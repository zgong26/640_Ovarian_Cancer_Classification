import csv
import random

import torchvision.models as models
import torch.nn as nn
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


def is_content_rich(square):
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


def process_image(img):
    res = []
    img_np = np.array(img)
    h, w, _ = img_np.shape
    square_size = int(np.sqrt(h * w / 1000))  # Calculate size of each square

    # Iterate over the image and slice it into squares
    for y in range(0, h - square_size + 1, square_size):
        for x in range(0, w - square_size + 1, square_size):
            square = img_np[y:y + square_size, x:x + square_size]
            if is_content_rich(square):
                res.append(square)

    # Randomly select 64 squares
    random.shuffle(res)
    return res[:min(64, len(res))]


def evaluate_image(model, squares):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    batch = []
    for square in squares:
        square_img = Image.fromarray(square)
        resized_img = square_img.resize((512, 512))
        tensor_img = transforms.ToTensor()(resized_img)
        normalized_tensor = normalize(tensor_img).to(device)
        batch.append(normalized_tensor)
    tensor_batch = torch.stack(batch).to(device)

    with torch.no_grad():
        output = model(tensor_batch)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)

    # Find the most common prediction
    pc_list = predicted_classes.tolist()
    most_common_prediction = max(set(pc_list), key=pc_list.count)
    confidences = [probabilities[i, most_common_prediction].item() for i in range(len(probabilities)) if
                   pc_list[i] == most_common_prediction]
    average_confidence = sum(confidences) / len(confidences) if confidences else 0
    print(f"confidence: {average_confidence}")
    return most_common_prediction


# Model Setup
Image.MAX_IMAGE_PIXELS = None
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)
model.load_state_dict(torch.load('ovarian_cancer_model_10_0.7943.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load and process the image
root_dir = 'test_images_compressed_80'
# img_list = [9,39,11,93,2]
img_list = list(range(108))
labels = ['CC', 'EC', 'LGSC', 'HGSC', 'MC']
for i in img_list:
    img_name = os.path.join(root_dir, f"{i}.jpg")
    image = Image.open(img_name).convert("RGB")
    squares = process_image(image)

    # Evaluate the image
    try:
        final_prediction = evaluate_image(model, squares)
        print(f"{i}: {labels[final_prediction]}")
    except:
        print(f"{i}: error")


"""
'CC': 0, 'EC': 1, 'LGSC': 2, 'HGSC': 3, 'MC': 4
"""
