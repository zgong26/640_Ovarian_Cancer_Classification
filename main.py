import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import torchvision.models as models
import os
from tqdm import tqdm


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
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Modify ConvNeXt Model
"""
# convnext model
model = models.convnext_base(pretrained=True)
model.classifier[2] = nn.Linear(in_features=1024, out_features=5)
"""
# ResNet-101 model
model = models.resnet101(pretrained=True)  # Load the pretrained ResNet-50 model
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)  # Replace the fully-connected layer with a new one with 5 outputs


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_accuracy = 0.0
# Training Loop
num_epochs = 99999999999999999
for epoch in range(num_epochs):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_predictions
    if epoch_accuracy >= max_accuracy:
        torch.save(model.state_dict(), 'ovarian_cancer_model.pth')
        max_accuracy = epoch_accuracy
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, MaxAccuracy: {max_accuracy:.4f}')

# Save the model checkpoint
torch.save(model.state_dict(), 'ovarian_cancer_model.pth')
