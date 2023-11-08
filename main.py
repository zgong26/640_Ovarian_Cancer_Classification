import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import torchvision.models as models
import os
from tqdm import tqdm

# Custom Dataset
"""
class OvarianCancerDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_dict = {'CC': 0, 'EC': 1, 'LGSC': 2, 'HGSC': 3, 'MC': 4}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]
        img_name = os.path.join(self.root_dir, f"{img_id}.jpg")
        Image.MAX_IMAGE_PIXELS = None
        image = Image.open(img_name).convert("RGB")
        label = self.annotations.iloc[index, 1]
        label = self.label_to_idx(label)  # Convert label to index

        if self.transform:
            image = self.transform(image)
        print(f"img: {img_id}, {self.annotations.iloc[index, 1]}")
        return image, label

    def label_to_idx(self, label):
        return self.label_dict[label]


# Dataset and DataLoader

dataset = OvarianCancerDataset(csv_file='train.csv',
                               root_dir='train_images_compressed_80',
                               transform=transform)

train_loader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)
"""


class OvarianCancerDataset(Dataset):
    def __init__(self, annotations_file, root_dir):
        self.annotations = pd.read_csv(annotations_file)
        self.root_dir = root_dir
        self.image_tensors = []
        self.labels = []
        self.preload_dataset()

    def preload_dataset(self):
        print("Preloading dataset to RAM...")
        for index in range(len(self.annotations)):
            img_id = self.annotations.iloc[index, 0]
            label = self.label_to_idx(self.annotations.iloc[index, 1])
            tensor_path = os.path.join(self.root_dir, f"{img_id}.pt")
            image_tensor = torch.load(tensor_path)
            self.image_tensors.append((image_tensor, label))
        print("Dataset preloaded.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Return preloaded tensors
        return self.image_tensors[index]

    def label_to_idx(self, label):
        label_dict = {'CC': 0, 'EC': 1, 'LGSC': 2, 'HGSC': 3, 'MC': 4}
        return label_dict.get(label, -1)  # Return -1 or any flag for unknown labels


# Usage
dataset = OvarianCancerDataset(annotations_file='train.csv', root_dir='modified_images_tensors')
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

# Modify ConvNeXt Model
model = models.convnext_base(pretrained=True)
model.classifier[2] = nn.Linear(in_features=1024, out_features=5)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training Loop
num_epochs = 99999999999999999
for epoch in range(num_epochs):
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), 'ovarian_cancer_model.pth')
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model checkpoint
torch.save(model.state_dict(), 'ovarian_cancer_model.pth')
