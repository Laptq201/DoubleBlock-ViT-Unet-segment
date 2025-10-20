import torch 
from dotenv import load_dotenv
from datetime import datetime
import wandb
import torch
from monai.transforms import (
    SpatialCrop,
    Activations,
    AsDiscrete,
    Compose,
    EnsureType
)
from loss import AverageMeter
from monai.inferers import sliding_window_inference
from brats import get_datasets
from .models.DB_MaxViT import model
from loss import EDiceLoss, EDiceLoss_Val
from monai.metrics import DiceMetric
import os
import numpy as np
from monai.data import decollate_batch

load_dotenv() # This loads the variables from .env into the environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
post_trans = Compose(
    [EnsureType(), Activations(sigmoid=True), AsDiscrete(argmax=False, threshold=0.5)]
)

post_sigmoid = Activations(sigmoid=True)
post_pred = AsDiscrete(argmax=False, threshold=0.5)
roi = (128, 128, 128) #128, 128, 128
sw_batch_size = 1
overlap = 0.5
VAL_AMP = True

wandb.login(key = WANDB_KEY)


def get_post_trans(tta=False):
    if tta:
        return Compose([
            EnsureType(),
            AsDiscrete(argmax=False, threshold=0.5)
        ])
    return Compose([
        EnsureType(),
        Activations(sigmoid=True),
        AsDiscrete(argmax=False, threshold=0.5)
    ])

#----------------
def train_epoch(model, loader, optimizer, epoch, loss_func, batch_size, wandb_tracking = 0):
    model.train()
    run_loss = AverageMeter('Loss', ':.4e')

    num_steps = len(loader)
    print(f"Epoch {epoch}: Number of steps (batches) in this epoch: {num_steps}")

    for idx, batch_data in enumerate(loader):
        torch.cuda.empty_cache()
        data, target = batch_data["image"].float().cuda(), batch_data["label"].float().cuda()
        logits = model(data)

        loss = loss_func(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        run_loss.update(loss.item(), n=batch_size)
        if wandb_tracking == 1:
            wandb.log({"train_step_loss": loss.item()})

    # Log the average loss for the epoch
    if wandb_tracking == 1:
        wandb.log({"train_epoch_loss": run_loss.avg, "epoch": epoch})
    return run_loss.avg

# ===============================================================================================
def model_inferer(input, model, tta=False, batch_size=2, overlap=0.6):
    def _compute(input, flip_dims=None):
        inputs = input if flip_dims is None else input.flip(dims=flip_dims)
        with torch.amp.autocast(device_type='cuda', enabled=VAL_AMP):
            output = sliding_window_inference(
                inputs=inputs,
                roi_size=roi,
                sw_batch_size=batch_size,
                predictor=model,
                overlap=overlap,
            )
        if flip_dims is not None:
            output = output.flip(dims=flip_dims)
        return output

    with torch.no_grad():
        if tta:
            # Danh sách các phép flip
            flip_combinations = [
                None,  # Gốc
                (2,),  # Flip x
                (3,),  # Flip y
                (4,),  # Flip z
                (2, 3),  # Flip x, y
                (2, 4),  # Flip x, z
                (3, 4),  # Flip y, z
                (2, 3, 4),  # Flip x, y, z
            ]
            predict = None
            for flip_dims in flip_combinations:
                output = _compute(input, flip_dims)
                output = torch.sigmoid(output)  # Chuyển logits thành xác suất
                predict = output if predict is None else predict + output
            predict = predict / 8.0  # Trung bình
        else:
            predict = _compute(input)  # Inference gốc, trả về logits

    torch.cuda.empty_cache()
    return predict


# ===============================================================================================

def evaluate_model(model, loader, epoch, acc_func, criterian_val, metric, wandb_tracking=0, tta=False):
    model.eval()
    run_acc = AverageMeter('Loss', ':.4e')
    all_preds = []
    all_labels = []
    post_trans = get_post_trans(tta)  # Chọn post_trans dựa trên tta
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            val_inputs, val_labels = batch_data["image"].to(device), batch_data["label"].to(device)

            logits = model_inferer(val_inputs, model, tta=tta, batch_size=2, overlap=0.6)
            val_outputs_list = decollate_batch(logits)
            val_labels_list = decollate_batch(val_labels)

            # Áp dụng post_trans
            val_output_convert = [post_trans(val_pred_tensor).to(device) for val_pred_tensor in val_outputs_list]
            val_labels_convert = [label.to(device) for label in val_labels_list]
            del val_inputs, val_labels, logits, val_outputs_list, val_labels_list
            torch.cuda.empty_cache()
            all_preds.extend(val_output_convert)
            all_labels.extend(val_labels_convert)

            # Tính metrics
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

        dice_et = run_acc.avg[0]
        dice_tc = run_acc.avg[1]
        dice_wt = run_acc.avg[2]

    # Log validation metrics
    if wandb_tracking == 1:
        wandb.log({
            "val_epoch_dice_et": dice_et,
            "val_epoch_dice_tc": dice_tc,
            "val_epoch_dice_wt": dice_wt,
            "epoch": epoch
        })
    return run_acc.avg

# ===============================================================================================


def trainer(model, train_loader, val_loader, optimizer, loss_func, 
            acc_func, criterian_val, metric, scheduler, tta=True, start_epoch=1, end_epoch=3, save_every=1, checkpoint_path=None, wandb_tracking = 0):
    # Initialize wandb logging
    if wandb_tracking:
        wandb.init(entity="uit-meow", project="medical", name="test-med1", config={
            "epochs": end_epoch,
            "optimizer": optimizer.__class__.__name__,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "loss_func": loss_func.__class__.__name__,
            "scheduler": scheduler.__class__.__name__ if scheduler else None
        })
        wandb_table = wandb.Table(columns=["Epoch", "Train Loss", "Dice_ET", "Dice_TC", "Dice_WT", "val_avg_acc", "Best Model"])

    val_acc_max = 0.0
    best_epoch = 0
    TC_dices = []
    WT_dices = []
    ET_dices = []
    avg_dices = []
    loss_epochs, train_epochs = [], []

    # If checkpoint path is provided, load model and optimizer states
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        val_acc_max = checkpoint['val_acc_max']  # Retain the best validation accuracy

        print(f"Resuming training from epoch {start_epoch}...")
    
    for epoch in range(start_epoch, end_epoch + 1):
        # Train the model for the current epoch
        train_loss = train_epoch(model, train_loader, optimizer, epoch=epoch, loss_func=loss_func, end=end_epoch, batch_size=1, wandb_tracking = wandb_tracking)
        torch.cuda.empty_cache()

        # Update scheduler if available
        if scheduler is not None:
            scheduler.step()
        
        # Evaluate the model after every 'save_every' epoch or at the last epoch
        if epoch % save_every == 0 or epoch == end_epoch or epoch == start_epoch:
            loss_epochs.append(train_loss)
            train_epochs.append(epoch)
            val_acc = evaluate_model(model, val_loader, epoch=epoch,
                                    acc_func=acc_func, criterian_val=criterian_val, 
                                    metric=metric, wandb_tracking=wandb_tracking, tta=tta)
            ET_dice = val_acc[0]
            TC_dice = val_acc[1]
            WT_dice = val_acc[2]
            val_avg_acc = np.mean(val_acc)

            # Update dice coefficients
            ET_dices.append(ET_dice)
            TC_dices.append(TC_dice)
            WT_dices.append(WT_dice)
            avg_dices.append(val_avg_acc)

            # Check if this is the best model
            best_model_flag = False
            if val_avg_acc > val_acc_max:
                print(f"New best ({val_acc_max:.6f} --> {val_avg_acc:.6f}) at epoch {epoch}")
                val_acc_max = val_avg_acc
                best_epoch = epoch
                best_model_flag = True
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
                'val_acc_max': val_acc_max,
            }, save_current)  # Save model to 'model.pth'
            wandb_table.add_data(epoch, train_loss, ET_dice, TC_dice, WT_dice, val_avg_acc, "Yes" if best_model_flag else "No")
            if best_model_flag:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
                    'val_acc_max': val_acc_max,
                }, save_path)  # Save model to 'model.pth'
                

            torch.cuda.empty_cache()

                

    # ✅ Log table to wandb
    if wandb_tracking == 1:
        wandb.log({"Training Metrics": wandb_table})

    return (val_acc_max, TC_dices, WT_dices, ET_dices, avg_dices, loss_epochs, train_epochs)


start = 0
end = 200#max_epochs
save_every = 2 #

learning_rate = 3e-4
weight_decay = 1e-5
checkpoint_path = ""
save_dir = "/kaggle/working"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "best_metric_model.pth")
save_current = os.path.join(save_dir, "current_model.pth")
###

criterion = EDiceLoss().cuda()
criterian_val = EDiceLoss_Val().cuda()
metric = criterian_val.metric

dice_acc = DiceMetric(include_background=True, reduction='mean_batch', get_not_nans=True)

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = end)



full_train_dataset, val_dataset = get_datasets(123, fold_number=0)
print(len(full_train_dataset), len(val_dataset))


"""
Create loader for neural network
"""


train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=1, shuffle=True,
                                           num_workers=2, pin_memory=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                         pin_memory=True, num_workers=2)

print("Train dataset number of batch:", len(train_loader))
print("Val dataset number of batch:", len(val_loader))

(val_acc_max, TC_dices, WT_dices, ET_dices, avg_dices, loss_epochs, train_epochs)  = trainer(model = model,
                                                                                            train_loader = train_loader,
                                                                                            val_loader = val_loader,
                                                                                            optimizer = optimizer,
                                                                                            loss_func = criterion,
                                                                                            acc_func = dice_acc,
                                                                                            criterian_val = criterian_val,
                                                                                            metric = metric,
                                                                                            scheduler = scheduler,
                                                                                            start_epoch = start,
                                                                                            end_epoch = end,
                                                                                            save_every = 2,
                                                                                            checkpoint_path = checkpoint_path,
                                                                                            wandb_tracking = 1)