# DoubleBlock-ViT: A MaxViT-Based Enhancement and Dual-Path Skip Connections for Brain Tumor Segmentation in MRI Scans

## Overview
This repository contains the official implementation of our paper  
**[DoubleBlock-ViT: A MaxViT-Based Enhancement and Dual-Path Skip Connections for Brain Tumor Segmentation in MRI Scans](https://github.com/Laptq201/DoubleBlock-ViT-Unet-segment)**  

### Authors
- **Thien B. Nguyen-Tat**  
- **Lap Quang Truong**  
- **Khanh Quoc Truong**  
- **Tai Huy Ngo**  
*(University of Information Technology, Vietnam National University, Ho Chi Minh City, Vietnam)*

---

## Pretrained Weights
You can download our pretrained model checkpoints from Google Drive:  
🔗 *[Download Weights (DoubleBlock-ViT-UNet, BraTS2021)](https://drive.google.com/drive/folders/1obXWQ9pa8QWhre-wgKt4JZGlhwYCO74t?usp=sharing)*  

---

## Abstract
Manual delineation of brain tumors from MRI scans is labor-intensive, error-prone, and requires extensive medical expertise.  
To address these challenges, this study introduces **DoubleBlock-ViT**, a lightweight hybrid 3D U-Net model that integrates **CNNs and Vision Transformers** to effectively capture both local and global contextual features.

The proposed framework leverages:
- A **DoubleBlock-ViT encoder**, which extends the MaxViT architecture for 3D volumetric MRI data.
- A **Dual-Path Fusion (DPF) block**, enhancing skip connections for improved feature refinement and semantic–spatial consistency.
- A **Project-and-Excite (PE)** mechanism to preserve spatial information and strengthen attention to clinically relevant regions.

Evaluations on **BraTS2020** and **BraTS2021** demonstrate state-of-the-art segmentation performance with only **7.8 million parameters**, showing both efficiency and accuracy suitable for clinical integration.

---

## Key Contributions
The proposed DoubleBlock-ViT U-Net introduces several innovations over conventional 3D U-Net and Transformer-based architectures:

1. 🧠 **Lightweight Hybrid Architecture**  
   A redesigned MaxViT-inspired encoder using **DoubleBlock-ViT** modules with positional encoding, MBConv, and localized attention — balancing accuracy and computational efficiency.

2. 🔀 **Dual-Path Fusion (DPF) Skip Connections**  
   A novel dual-path attention mechanism that adaptively fuses encoder–decoder features through dynamic gating, improving boundary delineation and context preservation.

3. ⚙️ **Improved Generalization and Scalability**  
   Achieves competitive performance with **7.8M parameters**, outperforming larger models like Swin UNETR and CKD-TransBTS, while reducing GPU memory consumption.

4. 🧩 **Robust 3D Segmentation Performance**  
   Demonstrated strong Dice and HD95 scores on BraTS2020 and BraTS2021 benchmarks:
   - **BraTS2020:** ET 80.11%, TC 86.60%, WT 91.20%  
   - **BraTS2021:** ET 87.82%, TC 91.61%, WT 92.31%

---

## Dataset
We use the **BraTS2020** and **BraTS2021** datasets, which provide multimodal MRI volumes of brain tumors with expert-annotated ground truth.

| Dataset | #Cases | Modalities | Resolution | Labels |
|----------|---------|-------------|-------------|---------|
| BraTS2020 | 369 | T1, T1Gd, T2, FLAIR | 240×240×155 | 4 |
| BraTS2021 | 1251 | T1, T1Gd, T2, FLAIR | 240×240×155 | 4 |

Each case includes four co-registered MRI modalities and voxel-wise annotations for:
- **ET (Enhancing Tumor)**
- **TC (Tumor Core)**
- **WT (Whole Tumor)**

Datasets are available at:  
- [BraTS 2020 - CBICA](https://www.med.upenn.edu/cbica/brats2020/data.html)  
- [BraTS 2021 - CBICA](https://www.med.upenn.edu/cbica/brats2021/data.html)

---

## Preprocessing
All MRI volumes are:
1. **Skull-stripped**, **co-registered**, and **resampled** to 1 mm³ isotropic resolution.  
2. **Cropped** to 128×128×128 using bounding boxes around tumor regions.  
3. **Intensity normalized** per modality via MONAI transforms.  
4. **Augmented** during training:
   - Random flipping (axial/sagittal/coronal)
   - Random cropping and scaling
   - Random intensity perturbations

---

## Model Architecture
The proposed **DoubleBlock-ViT U-Net** consists of three core modules:

1. **DoubleBlock-ViT Encoder**  
   - Combines 3D MBConv + Project-and-Excite for local context.  
   - Uses block-wise attention for long-range dependency modeling.  
   - Functions as a *conditional positional encoding* for volumetric MRI.

2. **Dual-Path Fusion (DPF) Block**  
   - Parallel refinement of encoder and decoder streams.  
   - Learnable gates via softmax–sigmoid to enhance feature interaction.  
   - Improves semantic–spatial consistency.

3. **Decoder and Output Layers**  
   - Uses lightweight 3D convolutions with GroupNorm and GELU.  
   - Output segmentation masks for ET, TC, WT classes.

---

## Training Process
1. Launch training from `main.py` or `main.ipynb`.  
2. Configure paths to datasets, checkpoints, and output directories.  
3. Modify model parameters (e.g., number of DB-ViT blocks or window size).  
4. Training setup:
   - Optimizer: **AdamW**
   - Learning rate: `3e-4`
   - Weight decay: `1e-5`
   - Epochs: `200`
   - Loss: **Hybrid Dice + Cross-Entropy**
   - Hardware: NVIDIA Tesla P100 GPU

---

## Installation
- You should use notebook file `notebook_DBMaxViT.ipynb` for easy setup and training.

## Pretrained Weights
You can download our pretrained model checkpoints from Google Drive:  
🔗 **[Download Weights (DoubleBlock-ViT-UNet, BraTS2021)](YOUR_DRIVE_LINK_HERE)**  

---
## Results

| Dataset | ET (Dice) | TC (Dice) | WT (Dice) | Params (M) |
|----------|------------|------------|------------|-------------|
| BraTS2020 | 80.11 | 86.60 | 91.20 | 7.8 |
| BraTS2021 | 87.82 | 91.61 | 92.31 | 7.8 |

Compared with state-of-the-art (SOTA) methods:

| Model | Year | Avg Dice | Params (M) |
|--------|------|-----------|------------|
| DynUNet | 2021 | 88.52 | 20+ |
| Swin UNETR | 2022 | 88.89 | 24 |
| CKD-TransBTS | 2023 | 87.36 | 22 |
| **DoubleBlock-ViT (Ours)** | **2025** | **90.58** | **7.8** |

---

## Visualization
- See in the paper

3D renderings demonstrate precise delineation of **Enhancing Tumor (ET)**, **Tumor Core (TC)**, and **Whole Tumor (WT)** regions, showing improved boundary accuracy compared to baseline models.


---

## References
1. [MaxViT: Multi-Axis Vision Transformer](https://arxiv.org/abs/2204.01697)  
2. [Project & Excite Module](https://arxiv.org/abs/1807.06521)  
3. [BraTS Challenge Overview](https://www.med.upenn.edu/cbica/brats2021/data.html)

---

## Citation
If you use this repository or paper, please cite:

```bibtex
@article{nguyen2025doubleblockvit,
  title={DoubleBlock-ViT: A MaxViT-Based Enhancement and Dual-Path Skip Connections for Brain Tumor Segmentation in MRI Scans},
  author={Nguyen-Tat, Thien B. and Truong, Lap Quang and Truong, Khanh Quoc and Ngo, Tai Huy},
  journal={Elsevier Preprint},
  year={2025},
  url={https://github.com/Laptq201/DoubleBlock-ViT-Unet-segment}
}