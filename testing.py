import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import torchvision.models as models
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

import os
import numpy as np
from tqdm import tqdm
import cv2




class SmartCenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def calculate_purple_score(self, hsv_square):
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

transform_pipeline = Compose([
    SmartCenterCrop(1024),
    # MakeSquare(fill_color=(0, 0, 0)),
    Resize((512, 512)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


model = models.resnet101(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)
model.load_state_dict(torch.load(
    'ovarian_cancer_model.pth'))  # Make sure 'ovarian_cancer_model.pth' locates at the same path as the script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

annotations = pd.read_csv('test.csv')
root_dir = 'test_images_compressed_80'
Image.MAX_IMAGE_PIXELS = None

for index, row in annotations.iterrows():
    img_id = row[0]
    img_name = os.path.join(root_dir, f"{img_id}.jpg")
    with Image.open(img_name).convert("RGB") as image:
        transformed_image = transform_pipeline(image).unsqueeze(0)
        transformed_image = transformed_image.to(device)
    with torch.no_grad():
        outputs = model(transformed_image)

    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    print(probabilities)

"""
'CC': 0, 'EC': 1, 'LGSC': 2, 'HGSC': 3, 'MC': 4
"""