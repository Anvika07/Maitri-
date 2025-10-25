import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# -----------------------------
# 1️⃣ Dataset class
# -----------------------------
class FatigueDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)

        # Handle missing data
        df = df.dropna(subset=['fatigue', 'tilt_angle', 'blinks', 'yawns'])

        # Encode fatigue labels (Tired / Alert / Drowsy → 0, 1, 2)
        le = LabelEncoder()
        df['fatigue_encoded'] = le.fit_transform(df['fatigue'])

        # Optional: print mapping for transparency
        print("[INFO] Fatigue label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

        # Features and labels
        X = df[['tilt_angle', 'blinks', 'yawns']].astype(float).values
        y = df['fatigue_encoded'].astype(int).values

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -----------------------------
# 2️⃣ Model
# -----------------------------
class FatigueModel(nn.Module):
    def __init__(self):
        super(FatigueModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),  # 3 classes (Tired, Alert, Drowsy)
        )

    def forward(self, x):
        return self.fc(x)

# -----------------------------
# 3️⃣ Training Function
# -----------------------------
def train_fatigue_model(csv_file="fatigue_data.csv", epochs=30, batch_size=8, lr=0.001):
    dataset = FatigueDataset(csv_file)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FatigueModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("[INFO] Starting training...")

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "fatigue_model.pth")
    print("[INFO] Training complete. Model saved as fatigue_model.pth")

# -----------------------------
# 4️⃣ Run Training
# -----------------------------
if __name__ == "__main__":
    train_fatigue_model()
