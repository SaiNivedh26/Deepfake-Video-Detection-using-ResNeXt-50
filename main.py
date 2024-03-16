# Import necessary libraries
import torch
import torchvision
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the DeepFake detection model
class DeepFakeDetector(nn.Module):
    def __init__(self, resnext_model, lstm_hidden_dim, num_classes):
        super(DeepFakeDetector, self).__init__()
        self.resnext_model = resnext_model
        self.lstm = nn.LSTM(input_size=resnext_model.fc.in_features, hidden_size=lstm_hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, num_classes)
    
    def forward(self, x):
        # Extract features using ResNeXt
        with torch.no_grad():
            features = self.resnext_model(x)
        # LSTM input shape: (batch_size, seq_len, input_size)
        features = features.unsqueeze(1)  # Add a dimension for sequence length
        # Pass features through LSTM
        lstm_out, _ = self.lstm(features)
        # Take output from last time step
        lstm_out = lstm_out[:, -1, :]
        # Pass through fully connected layer
        output = self.fc(lstm_out)
        return output

# Define the dataset class
class DeepFakeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = os.listdir(data_dir)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = self.load_images()

    def load_images(self):
        images = []
        for cls in self.classes:
            cls_dir = os.path.join(self.data_dir, cls)
            for img_file in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_file)
                images.append((img_path, self.class_to_idx[cls]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = torchvision.io.read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label

# Set up data transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define parameters
data_dir = 'path/to/FFHQ_dataset'
batch_size = 32
num_classes = 2
lstm_hidden_dim = 512

# Create dataset and dataloaders
dataset = DeepFakeDataset(data_dir, transform=data_transforms)
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Load ResNeXt model from Hugging Face
resnext_model = timm.create_model("hf_hub:timm/resnext101_32x16d.fb_swsl_ig1b_ft_in1k", pretrained=True)
# Freeze parameters
for param in resnext_model.parameters():
    param.requires_grad = False
# Modify the last fully connected layer to match the number of classes
resnext_model.fc = nn.Linear(resnext_model.fc.in_features, lstm_hidden_dim)

# Initialize the DeepFake detector model
model = DeepFakeDetector(resnext_model, lstm_hidden_dim, num_classes)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")
