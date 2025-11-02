import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


# Example ST-GCN model skeleton, replace with full implementation or import
class SimpleSTGCN(nn.Module):
    def __init__(self, num_joints, in_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(num_joints * in_channels, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, num_joints, in_channels)
        # Flatten joints and channels
        x = x.view(x.size(0), x.size(1), -1)  # (batch, seq_len, features)
        x = x.mean(dim=1)  # simple temporal pooling for this example
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Dataset prototype: load keypoint json sequence and label
class PoseDataset(Dataset):
    def __init__(self, root_folder, classes, seq_len=30):
        self.samples = []
        self.classes = classes
        self.seq_len = seq_len
        for cls_idx, cls_name in enumerate(classes):
            cls_folder = os.path.join(root_folder, cls_name, 'keypoints')
            files = sorted(os.listdir(cls_folder))
            # For simplicity, assume the dataset is split into sequences of length seq_len
            for i in range(0, len(files) - seq_len + 1, seq_len):
                seq_files = files[i:i + seq_len]
                self.samples.append(([os.path.join(cls_folder, f) for f in seq_files], cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        files, label = self.samples[idx]
        seq_keypoints = []
        for f in files:
            with open(f, 'r') as jf:
                data = json.load(jf)
            # data is list of dicts with keys x,y,z,visibility
            # Convert to numpy array shape (num_joints, 3)
            kp_array = np.array([[pt['x'], pt['y'], pt['z']] for pt in data], dtype=np.float32)
            seq_keypoints.append(kp_array)
        seq_keypoints = np.stack(seq_keypoints)  # (seq_len, num_joints, 3)
        return torch.tensor(seq_keypoints), label


# Parameters
root_folder = r'C:\Users\apsal\OneDrive\Desktop\examhallai\humangbjhgh'  # contains standing/, bending/, etc.
classes = ['standing', 'bending', 'turningaround', 'normal']
num_classes = len(classes)
num_joints = 33  # depends on your keypoint format, e.g., 33 for MediaPipe pose
in_channels = 3  # x,y,z
seq_len = 30

# Create dataset and dataloader
dataset = PoseDataset(root_folder, classes, seq_len)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model, loss, optimizer
model = SimpleSTGCN(num_joints, in_channels, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop (basic)
num_epochs = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")

# Save the trained model
save_path = r'C:\Users\apsal\OneDrive\Desktop\examhallai\humangbjhgh\stgcn_model.pth'
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")

print("Training completed.")
