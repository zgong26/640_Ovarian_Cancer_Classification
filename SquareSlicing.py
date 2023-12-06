import csv

import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torchvision.transforms as transforms
import concurrent.futures
import cv2
import numpy as np
from PIL import Image, ImageOps
from queue import Queue


class SmartCrop(object):
    def __init__(self, num_squares=1000):
        self.num_squares = num_squares

    def is_content_rich(self, square):

        gray_square = cv2.cvtColor(square, cv2.COLOR_RGB2GRAY)

        # Calculate the percentage of black and white pixels
        black_and_white_content = np.sum((gray_square < 40) | (gray_square > 200))

        # Convert square to HSV for purple color check
        hsv_square = cv2.cvtColor(square, cv2.COLOR_RGB2HSV)

        # Define range for purple hue in HSV
        lower_purple = np.array([130, 50, 50])
        upper_purple = np.array([150, 255, 255])

        # Create a mask that captures areas within the purple range
        mask = cv2.inRange(hsv_square, lower_purple, upper_purple)

        # Calculate the percentage of purple pixels
        purple_content = np.sum(mask)

        # Decide if the content is rich based on black & white and purple content
        num_pixels = square.shape[0] * square.shape[1]
        black_and_white_percentage = (black_and_white_content / num_pixels) * 100
        purple_percentage = (purple_content / num_pixels) * 100
        return black_and_white_percentage < 50 and purple_percentage > 75

    def __call__(self, img):
        # Convert PIL Image to numpy array
        img_np = np.array(img)

        # Calculate the size of each square
        h, w, _ = img_np.shape
        square_size = int(np.sqrt(h * w / self.num_squares))
        # Define the normalization transform
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # Iterate over the image and slice it into squares
        for y in range(0, h, square_size):
            for x in range(0, w, square_size):
                square = img_np[y:y + square_size, x:x + square_size]

                # Check if the square is content-rich
                if self.is_content_rich(square):
                    if not tensor:
                        square_img = Image.fromarray(square)
                        resized_img = square_img.resize((512, 512))
                        yield resized_img
                    else:
                        square_img = Image.fromarray(square)
                        resized_img = square_img.resize((512, 512))
                        tensor_img = transforms.ToTensor()(resized_img)
                        normalized_tensor = normalize(tensor_img)
                        yield normalized_tensor


def transform(img_name, id, label):
    with Image.open(img_name).convert("RGB") as image:
        cropper = SmartCrop(num_squares=1000)
        subid = 1
        for content_square in cropper(image):
            square_filename = f"{id}_{subid}"
            if tensor:
                torch.save(content_square, os.path.join(output_dir, f"{square_filename}.pt"))
            else:
                content_square.save(os.path.join(output_dir, f"{square_filename}.jpg"))
            in_memory_data.put((square_filename, label))
            subid += 1


def process_image(row):
    img_id = row[0]
    img_name = os.path.join(root_dir, f"{img_id}.jpg")
    try:
        transform(img_name, img_id, row[1])
    except IOError as e:
        # Handle exceptions
        print(f"Error processing {img_id}: {e}")


def process_dataset_multithreaded(annotations, max_workers=16):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use a list comprehension to create a list of futures
        futures = [executor.submit(process_image, row) for index, row in annotations.iterrows()]
        # Use tqdm to display progress
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            continue


def save_to_csv(csv_file_path, data_queue):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_id', 'label'])  # Write the CSV header

        while not data_queue.empty():
            writer.writerow(data_queue.get())


# Usage
tensor = True  # this flag ensures whether tensors are saved or jpg images are saved
Image.MAX_IMAGE_PIXELS = None
root_dir = 'train_images_compressed_80'  # Original images directory
output_dir = 'modified_images_tensors'  # Directory where transformed tensors will be saved
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

"""
img_id = 4
img_name = os.path.join(root_dir, f"{img_id}.jpg")
with Image.open(img_name).convert("RGB") as image:
    cropper = SmartCrop(num_squares=1000)
    for content_square in cropper(image):
        content_square.save(output_dir)
"""
annotations = pd.read_csv('train.csv')
in_memory_data = Queue()
process_dataset_multithreaded(annotations)
csv_file_path = 'new_train.csv'
save_to_csv(csv_file_path, in_memory_data)
