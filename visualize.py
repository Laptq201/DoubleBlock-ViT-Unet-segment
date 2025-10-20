

import matplotlib.pyplot as plt
import numpy as np

def visualize_samples(data_loader):
    
    label_colors = ['red','yellow', 'green']
    color_values = {'red': (255, 0, 0, 255),  'yellow': (255, 255, 0, 255),'green': (0, 255, 0, 255)}

    background_color = (0, 0, 0, 255)
    i=0
    # Get a single batch from the DataLoader
    for batch in data_loader:

        print(batch['patient_id'])
        image_sample = batch['image']
        label_sample = batch['label']
        image_sample_np = image_sample.numpy()
        label_sample_np = label_sample.numpy()


        z_slice =  image_sample_np.shape[2] // 2 +2 # between

        # Plot each channel of the image sample
        num_channels = image_sample_np.shape[1]
        fig, axes = plt.subplots(1, num_channels, figsize=(15, 5))

        for channel in range(num_channels):
            axes[channel].imshow(image_sample_np[0, channel, z_slice], cmap='gray')
            axes[channel].set_title(f"Image Channel {channel + 1}")


        plt.tight_layout()
        plt.show()

        # Plot each channel of the label sample (assuming 3 channels for one-hot encoded segmentation)
        num_channels_labels = label_sample_np.shape[1]
        fig, axes = plt.subplots(1, num_channels_labels, figsize=(15, 5))

        for channel in range(num_channels_labels):
            axes[channel].imshow(label_sample_np[0, channel, z_slice], cmap='gray')
            axes[channel].set_title(f"Label Channel {channel + 1}")


        plt.tight_layout()
        plt.show()

        image_sample_np = np.full((label_sample_np.shape[3], label_sample_np.shape[4], 4), background_color, dtype=np.uint8)
        # Combine the label channels with different colors
        num_channels_labels = label_sample_np.shape[1]
        for channel in range(num_channels_labels-1, -1, -1):
            label_channel = label_sample_np[0, channel, z_slice]

            # Overlay the label with a unique color
            label_color = label_colors[channel % len(label_colors)]
            color_value = color_values[label_color]

            # Create a mask for the current label channel
            label_mask = label_channel > 0

            # Apply the color with alpha channel to the corresponding pixels in the composite label
            image_sample_np[label_mask] = color_value

        # Plot the composite label image
        plt.figure(figsize=(10, 5))
        plt.imshow(image_sample_np)
        plt.title("Composite Label")
        plt.show()
        i+=1
        if i == 10:
            break