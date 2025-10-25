import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn, optim

# -------------------------
# Dataset class
# -------------------------
class NTHUDDDDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: path to train_data folder containing 'drowsy' and 'notdrowsy'
        """
        self.samples = []
        self.transform = transform
        classes = ['notdrowsy', 'drowsy']

        for label, folder in enumerate(classes):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Folder not found: {folder_path}")
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(folder_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# -------------------------
# Simple CNN for binary classification
# -------------------------
class FatigueCNN(nn.Module):
    def __init__(self):
        super(FatigueCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*32*32, 128), nn.ReLU(),
            nn.Linear(128, 2)  # 2 classes: notdrowsy / drowsy
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# -------------------------
# Training function
# -------------------------
def train_model(root_dir, epochs=5, batch_size=32, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])
    dataset = NTHUDDDDataset(root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = FatigueCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f} | Accuracy: {correct/len(dataset):.4f}")

    torch.save(model.state_dict(), "fatigue_model_nthuddd.pth")
    print("[INFO] Training complete. Model saved as 'fatigue_model_nthuddd.pth'")

# -------------------------
# Main
# -------------------------
if __name__=="__main__":
    dataset_root = r"D:\Projects\Physical\datasets\nthuddd2\train_data"  # change if needed
    train_model(dataset_root, epochs=5)
