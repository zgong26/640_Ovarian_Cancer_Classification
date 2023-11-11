import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image, ImageOps
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import cv2
import random
import cv2
import numpy as np
from PIL import Image


class SmartCenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def calculate_purple_score(self, hsv_square):
        # Define range for purple hue in HSV
        lower_purple = np.array([130, 50, 50])  # Adjust as needed
        upper_purple = np.array([150, 255, 255])  # Adjust as needed

        # Create a mask that captures areas within the purple range
        mask = cv2.inRange(hsv_square, lower_purple, upper_purple)

        # The score is the sum of all pixels in the mask
        score = np.sum(mask)
        return score

    def __call__(self, img):
        # Convert PIL Image to a numpy array
        img_np = np.array(img)

        # Convert to HSV color space
        hsv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

        # Convert to grayscale for contour detection
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Apply threshold to get the binary image
        _, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If no contours are found, return the original image
        if not contours:
            return img

        # Find the largest contour and its bounding box
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Initialize best score and best square
        best_score = -1
        best_square = None

        # Attempt to find the best square within the largest contour
        for _ in range(6):
            # Randomly choose a starting point for the square
            random_x = x + np.random.randint(0, max(1, w - self.output_size))
            random_y = y + np.random.randint(0, max(1, h - self.output_size))

            # Extract the square from the HSV image
            hsv_square = hsv_img[random_y:random_y + self.output_size, random_x:random_x + self.output_size]

            # Calculate the score for the square based on purple content
            score = self.calculate_purple_score(hsv_square)

            # If the score is better than the best score, update the best square
            if score > best_score:
                best_score = score
                best_square = img_np[random_y:random_y + self.output_size, random_x:random_x + self.output_size]

        # Convert the best square numpy array back to PIL Image if a best square was found
        if best_square is not None:
            return Image.fromarray(best_square)
        else:
            return img  # Return the original if no best square was found



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
    SmartCenterCrop(1024),
    # MakeSquare(fill_color=(0, 0, 0)),
    Resize((512, 512)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_pipeline_image = Compose([
    SmartCenterCrop(1024),
    # MakeSquare(fill_color=(0, 0, 0)),
    Resize((512, 512)),
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
    except IOError as e:
        # Handle exceptions (e.g., file not found, etc.)
        print(f"Error processing {img_id}: {e}")


def process_dataset_multithreaded(annotations, max_workers=16):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use a list comprehension to create a list of futures
        futures = [executor.submit(process_image, row) for index, row in annotations.iterrows()]
        # Use tqdm to display progress
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            continue


# Read the CSV file containing image IDs and labels
annotations = pd.read_csv('train.csv')
root_dir = 'train_images_compressed_80'  # Original images directory
output_dir = 'modified_images_tensors'  # Directory where transformed tensors will be saved

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

process_dataset_multithreaded(annotations)

print("All image tensors have been processed and saved.")
