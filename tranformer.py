import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image, ImageOps
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

# Define a custom crop transform
class SmartCenterCrop(object):
    def __init__(self, output_size, threshold=50):
        self.output_size = output_size
        self.threshold = threshold

    def __call__(self, img):
        # Convert to grayscale and to a numpy array
        gray_img = ImageOps.grayscale(img)
        img_array = np.array(gray_img)

        # Calculate the center of the image
        center_x, center_y = img_array.shape[1] // 2, img_array.shape[0] // 2
        half_crop = self.output_size // 2

        # Check if the center region is mostly black
        center_region = img_array[center_y - half_crop:center_y + half_crop, center_x - half_crop:center_x + half_crop]
        if np.mean(center_region) < self.threshold:
            # If center is black, crop the center of the right half
            start_x = 3 * img_array.shape[1] // 4 - half_crop
            end_x = start_x + self.output_size
            img = img.crop((start_x, center_y - half_crop, end_x, center_y + half_crop))
        else:
            # Otherwise, perform a regular center crop
            img = CenterCrop(self.output_size)(img)

        return img


# Define your transformations
transform_pipeline = Compose([
    SmartCenterCrop(1024),
    Resize((256, 256)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Read the CSV file containing image IDs and labels
annotations = pd.read_csv('train.csv')
root_dir = 'train_images_compressed_80'  # Original images directory
output_dir = 'modified_images_tensors'  # Directory where transformed tensors will be saved

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Preprocess and save the images as tensors
for idx, row in tqdm(annotations.iterrows()):
    Image.MAX_IMAGE_PIXELS = None
    img_id = row[0]
    img_name = os.path.join(root_dir, f"{img_id}.jpg")
    image = Image.open(img_name).convert("RGB")

    # Apply transformations
    transformed_image = transform_pipeline(image)

    # Save the tensor to disk
    torch.save(transformed_image, os.path.join(output_dir, f"{img_id}.pt"))

print("All image tensors have been processed and saved.")
