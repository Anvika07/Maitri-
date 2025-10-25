import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# -------------------------------
# 1. Define architecture
# -------------------------------
class FatigueCNN(nn.Module):
    def __init__(self):
        super(FatigueCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# -------------------------------
# 2. Load model safely
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FatigueCNN().to(device)

# Load checkpoint
checkpoint = torch.load("fatigue_model_nthuddd.pth", map_location=device)

# Remap keys from fc.* → classifier.*
new_state_dict = {}
for key, value in checkpoint.items():
    if key.startswith("fc."):
        new_key = key.replace("fc.", "classifier.")
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

# Load the corrected state dict
model.load_state_dict(new_state_dict, strict=False)
model.eval()
print("✅ Model loaded successfully after remapping keys!")


# -------------------------------
# 3. Define preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# -------------------------------
# 4. Start webcam detection
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("❌ Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        confidence = torch.max(output).item()

    label = "😴 Drowsy" if pred == 1 else "😊 Alert"
    color = (0, 0, 255) if pred == 1 else (0, 255, 0)
    text = f"{label} ({confidence*100:.1f}%)"

    cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    cv2.imshow("Fatigue Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
