import timm
from PIL import Image
import torch
import onednn
from torchvision import transforms
import cv2
import os
import gradio as gr

# Data preprocessing
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img_size = 224

train_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

class CelebDFDataset(torch.utils.data.Dataset):
    def _init_(self, data_dir, transforms=None, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        self.transforms = transforms
        self.data = self.load_data()

    def load_data(self):
        data = []
        for label, folder_name in enumerate(['Celeb-real', 'Celeb-synthesis']):  # Corrected folder names
            label_dir = os.path.join(self.data_dir, folder_name)
            for img_path in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_path)
                data.append((img_path, label))
        return data

    def _len_(self):
        return len(self.data)

    def _getitem_(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        return img, label

# Set data paths
data_dir = 'directory to celeb df dataset folder'

# Load datasets
train_dataset = CelebDFDataset(data_dir, transforms=train_transforms, mode='train')
val_dataset = CelebDFDataset(data_dir, transforms=val_transforms, mode='val')

# Create data loaders
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load the model
device = torch.device("cpu")  # Use CPU for oneDNN
model = timm.create_model("resnext50_32x4d", pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)

def set_model_eval(model):
    for module in model.modules():
        if hasattr(module, 'training'):
            module.training = False

model.eval()
set_model_eval(model)
model = onednn.to_onednn(model)  # Convert model to use oneDNN
model = model.to(device)

# Function to preprocess and classify frames
def classify_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)  # Convert to PIL Image
        frame = val_transforms(frame).unsqueeze(0)
        frames.append(frame)

    if len(frames) == 0:
        return predictions  # Return empty predictions if no frames were read

    frames = torch.cat(frames, dim=0).to(device)
    with torch.no_grad():
        outputs = model(frames)
        _, preds = torch.max(outputs, 1)
        predictions = [bool(pred) for pred in preds.cpu().numpy()]

    cap.release()
    return predictions

# Function to determine final classification
def final_classification(predictions, threshold=0.5):
    fake_count = sum(predictions)
    total_frames = len(predictions)
    fake_percentage = fake_count / total_frames
    if fake_percentage >= threshold:
        return "Fake"
    else:
        return "Real"

def classify_video(video):
    if video is None:
        return "Please upload a video file."
    
    predictions = classify_frames(video)
    print("Predictions:", predictions)  # Debugging statement
    final_result = final_classification(predictions)
    return final_result

title = "Deepfake Detector"
description = "Made by Team Nooglers as a part of Intel AI hackathon 😎"

iface = gr.Interface(
    fn=classify_video,
    inputs=gr.Video(label="Upload Video"),
    outputs=gr.Label(label="Result"),
    title=title,
    description=description,
    examples=[["directory to example video"]]
)

iface.launch()
