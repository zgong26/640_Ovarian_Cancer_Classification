import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import torchvision.models as models
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

TRAIN_SUBSET_RATIO = 0.2  # 20% of the training data
VAL_SUBSET_RATIO = 0.2  # 20% of the validation data


def get_subset_indices(total, ratio):
    indices = np.arange(total)
    np.random.shuffle(indices)
    subset_size = int(total * ratio)
    return indices[:subset_size]


class OvarianCancerDataset(Dataset):
    def __init__(self, annotations_df, root_dir, indices=None):
        self.annotations = annotations_df if indices is None else annotations_df.iloc[indices]
        self.root_dir = root_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]
        label = self.label_to_idx(self.annotations.iloc[index, 1])
        tensor_path = os.path.join(self.root_dir, f"{img_id}.pt")
        image_tensor = torch.load(tensor_path)
        return image_tensor, label

    def label_to_idx(self, label):
        label_dict = {'CC': 0, 'EC': 1, 'LGSC': 2, 'HGSC': 3, 'MC': 4}
        return label_dict.get(label, -1)


# !!!!!
ep = 9

annotations = pd.read_csv('averaged_train.csv')
train_annotations, val_annotations = train_test_split(annotations, test_size=0.2, random_state=42)
train_dataset = OvarianCancerDataset(annotations_df=train_annotations, root_dir='modified_images_tensors')
val_dataset = OvarianCancerDataset(annotations_df=val_annotations, root_dir='modified_images_tensors')
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# ResNet-101 model
model = models.resnet50(pretrained=True)  # Load the pretrained ResNet-50 model
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)  # Replace the fully-connected layer with a new one with 5 outputs
# Load the saved state dict
model.load_state_dict(torch.load('ovarian_cancer_model_8_0.7342.pth'))

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_accuracy = 0.0
# Training Loop
num_epochs = 9999999
for epoch in range(ep - 1, num_epochs):
    # Create datasets for this epoch
    epoch_train_dataset = OvarianCancerDataset(annotations_df=train_annotations, root_dir='modified_images_tensors')
    epoch_val_dataset = OvarianCancerDataset(annotations_df=val_annotations, root_dir='modified_images_tensors')

    # Create DataLoaders for this epoch
    epoch_train_loader = DataLoader(epoch_train_dataset, batch_size=64, shuffle=True)
    epoch_val_loader = DataLoader(epoch_val_dataset, batch_size=64, shuffle=False)
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for images, labels in tqdm(epoch_train_loader):
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

    epoch_loss = running_loss / len(epoch_train_loader)
    epoch_accuracy = correct_predictions / total_predictions
    print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}')

    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in epoch_val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_epoch_loss = val_loss / len(epoch_val_loader)
    val_epoch_accuracy = val_correct / val_total
    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.4f}')

    torch.save(model.state_dict(), f'ovarian_cancer_model_{epoch + 1}_{val_epoch_accuracy:.4f}.pth')

# Save the model checkpoint
torch.save(model.state_dict(), 'ovarian_cancer_model.pth')
