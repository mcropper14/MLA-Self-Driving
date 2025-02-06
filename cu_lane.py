#cropthecoder/Downloads/driver_182_30frame-003/driver_182_30frame
#driver_182_30frame-003/driver_182_30frame



#https://github.com/XingangPan/SCNN
#https://xingangpan.github.io/projects/CULane.html


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
import torchvision.models as models
import os
from PIL import Image
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



class LaneDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.jpg')])
        self.label_files = [f.replace('.jpg', '.lines.txt') for f in self.image_files]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        label_path = os.path.join(self.data_dir, self.label_files[idx])

        image = Image.open(image_path).convert("RGB")

        with open(label_path, "r") as file:
            lines = file.readlines()
            lanes = []
            width, height = image.size
            for line in lines:
                points = line.strip().split()
                lane = [(float(points[i]) / width, float(points[i + 1]) / height) for i in range(0, len(points), 2)]
                lanes.append(lane)

        if self.transform:
            image = self.transform(image)

        # Pad lane data to a fixed size
        max_lanes = 5  # idk prob
        max_points = 32  # idk prob
        padded_lanes = []
        for lane in lanes[:max_lanes]:
            padded_lane = lane[:max_points] + [(0, 0)] * (max_points - len(lane))
            padded_lanes.append(padded_lane)
        # Pad extra lanes if fewer than max_lanes, if not it fucks shit up
        while len(padded_lanes) < max_lanes:
            padded_lanes.append([(0, 0)] * max_points)

        lanes_tensor = torch.tensor(padded_lanes, dtype=torch.float32)  # [max_lanes, max_points, 2]

        return image, lanes_tensor

# Define Multi-query Latent Attention Mechanism
# Straight up from github 
class MLA(nn.Module):
    def __init__(self, input_dim, latent_dim, num_heads):
        super(MLA, self).__init__()
        self.num_heads = num_heads
        self.latent_dim = latent_dim

        self.query_layer = nn.Linear(input_dim, latent_dim * num_heads)
        self.key_layer = nn.Linear(input_dim, latent_dim * num_heads)
        self.value_layer = nn.Linear(input_dim, latent_dim * num_heads)

        self.output_layer = nn.Linear(latent_dim * num_heads, input_dim)

    def forward(self, x):
        # Compute Q, K, V
        Q = self.query_layer(x)
        K = self.key_layer(x)
        V = self.value_layer(x)

        # Reshape for multi-head attention
        Q = Q.view(x.size(0), self.num_heads, self.latent_dim)
        K = K.view(x.size(0), self.num_heads, self.latent_dim)
        V = V.view(x.size(0), self.num_heads, self.latent_dim)

        # Compute attention scores
        scores = torch.einsum('bhd,bhd->bh', Q, K) / (self.latent_dim ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply attention weights to values
        attended_values = torch.einsum('bh,bhd->bhd', attention_weights, V)

        # Reshape back and pass through output layer
        attended_values = attended_values.view(x.size(0), -1)
        output = self.output_layer(attended_values)
        return output

#one fc is not enough 
# 2 is good ig
class LaneDetectionModel(nn.Module):
    def __init__(self, input_dim, latent_dim, num_heads, max_lanes, max_points):
        super(LaneDetectionModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove fully connected layer

        self.mla = MLA(input_dim, latent_dim, num_heads)
        self.fc1 = nn.Linear(input_dim, 1024)  # Add intermediate layers
        self.fc2 = nn.Linear(1024, max_lanes * max_points * 2)  # Predict coordinates

        self.activation = nn.ReLU()

    def forward(self, x):
        features = self.backbone(x)
        attention_output = self.mla(features)
        x = self.activation(self.fc1(attention_output))
        output = self.fc2(x)
        output = output.view(x.size(0), -1, 32, 2)  # Reshape to [batch, max_lanes, max_points, 2]
        return output

# Training Setup
def train_model(model, dataloader, criterion, optimizer, num_epochs=10): #15, 10
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, labels in dataloader:
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

def evaluate_model(model, dataloader):
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            all_predictions.append(outputs.numpy())
            all_targets.append(labels.numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    mse = mean_squared_error(all_targets.flatten(), all_predictions.flatten())
    mae = mean_absolute_error(all_targets.flatten(), all_predictions.flatten())

    r2 = r2_score(all_targets.flatten(), all_predictions.flatten())

    print(f"Evaluation Results: MSE = {mse:.4f}, MAE = {mae:.4f}, R^2 = {r2:.4f}")


if __name__ == "__main__":
    data_dir = "/home/cropthecoder/Downloads/driver_182_30frame-003/driver_182_30frame/05312327_0001.MP4"


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = LaneDataset(data_dir, transform=transform)

    #wtf why 
    print(f"Total dataset size: {len(full_dataset)}")
    print(f"Example image file: {full_dataset.image_files[0]}")
    print(f"Example label file: {full_dataset.label_files[0]}")

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    max_lanes = 5
    max_points = 32 #idk 
    model = LaneDetectionModel(input_dim=512, latent_dim=64, num_heads=8, max_lanes=max_lanes, max_points=max_points)
    #criterion = nn.MSELoss()
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=1e-4) #-4

    train_model(model, train_loader, criterion, optimizer)
    evaluate_model(model, test_loader)