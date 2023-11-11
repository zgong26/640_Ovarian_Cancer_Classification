import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image, ImageOps
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures


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


# Custom square padding transformation
class MakeSquare(object):
    def __init__(self, fill_color=(0, 0, 0)):
        self.fill_color = fill_color

    def __call__(self, image):
        x, y = image.size
        size = max(y, x)
        new_im = Image.new("RGB", (size, size), self.fill_color)
        new_im.paste(image, (int((size - x) / 2), int((size - y) / 2)))
        return new_im


# Define your transformations
transform_pipeline = Compose([
    SmartCenterCrop(512),
    # MakeSquare(fill_color=(0, 0, 0)),
    # Resize((512, 512)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_pipeline_image = Compose([
    SmartCenterCrop(512),
    # MakeSquare(fill_color=(0, 0, 0)),
    # Resize((512, 512)),
])


def process_image(row):
    tensor = True  # this flag ensures whether tensors are saved or jpg images are saved
    img_id = row[0]
    img_name = os.path.join(root_dir, f"{img_id}.jpg")

    try:
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(img_name).convert("RGB") as image:
            # Apply transformations
            if tensor:
                transformed_image = transform_pipeline(image)
            else:
                transformed_image = transform_pipeline_image(image)

        # Save the tensor to disk
        if tensor:
            torch.save(transformed_image, os.path.join(output_dir, f"{img_id}.pt"))
        else:
            transformed_image.save(os.path.join(output_dir, f"{img_id}.jpg"))
        return f"{img_id}.pt saved successfully."
    except IOError as e:
        # Handle exceptions (e.g., file not found, etc.)
        return f"Error processing {img_id}: {e}"


def process_dataset_multithreaded(annotations, max_workers=16):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use a list comprehension to create a list of futures
        futures = [executor.submit(process_image, row) for index, row in annotations.iterrows()]
        # Use tqdm to display progress
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            print(future.result())


# Read the CSV file containing image IDs and labels
annotations = pd.read_csv('train.csv')
root_dir = 'train_images_compressed_80'  # Original images directory
output_dir = 'modified_images_tensors'  # Directory where transformed tensors will be saved

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Preprocess and save the images as tensors
"""
for idx, row in tqdm(annotations.iterrows()):
    Image.MAX_IMAGE_PIXELS = None
    img_id = row[0]
    img_name = os.path.join(root_dir, f"{img_id}.jpg")
    image = Image.open(img_name).convert("RGB")

    # Apply transformations
    transformed_image = transform_pipeline(image)
    #transformed_image = transform_pipeline_image(image)

    # Save the tensor to disk
    torch.save(transformed_image, os.path.join(output_dir, f"{img_id}.pt"))
    #transformed_image.save(os.path.join(output_dir, f"{img_id}.jpg"))
"""
process_dataset_multithreaded(annotations)

print("All image tensors have been processed and saved.")
