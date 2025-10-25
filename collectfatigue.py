import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
import math
import os

# ──────────────────────────────
# SETTINGS
# ──────────────────────────────
FATIGUE_CSV = "fatigue_data.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────
# MODELS
# ──────────────────────────────
class FatigueNet(nn.Module):
    def __init__(self):
        super(FatigueNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(468 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        return self.fc(x)

class EmotionNet(nn.Module):
    def __init__(self):
        super(EmotionNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 12 * 12, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ──────────────────────────────
# INIT
# ──────────────────────────────
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
fatigue_model = FatigueNet().to(DEVICE)
emotion_model = EmotionNet().to(DEVICE)

if os.path.exists("emotion_model.pth"):
    emotion_model.load_state_dict(torch.load("emotion_model.pth", map_location=DEVICE))
    print("[INFO] Loaded pretrained emotion model.")

FATIGUE_LABELS = {0: "Alert", 1: "Tired", 2: "Drowsy"}
EMOTION_LABELS = {0: "Happy", 1: "Neutral", 2: "Sad"}

# ──────────────────────────────
# UTILS
# ──────────────────────────────
def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def eye_aspect_ratio(landmarks, eye_indices):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    A = euclidean(p2, p6)
    B = euclidean(p3, p5)
    C = euclidean(p1, p4)
    return (A + B) / (2.0 * C)

def mouth_opening_ratio(landmarks, mouth_indices):
    top, bottom = landmarks[mouth_indices[0]], landmarks[mouth_indices[1]]
    return euclidean(top, bottom)

def head_tilt_angle(landmarks):
    left_eye = np.mean([landmarks[i] for i in [33, 133]], axis=0)
    right_eye = np.mean([landmarks[i] for i in [362, 263]], axis=0)
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle

# ──────────────────────────────
# MAIN PROCESS
# ──────────────────────────────
def auto_collect_advanced():
    cap = cv2.VideoCapture(0)
    data = []
    blink_count, yawn_count = 0, 0
    prev_ear, prev_mor = 0, 0
    blink_threshold = 0.20
    yawn_threshold = 0.06

    print("[INFO] Starting advanced monitoring — press ESC to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face_mesh.process(rgb)
        fatigue_label = "Unknown"
        emotion_label = "Unknown"

        if res.multi_face_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in res.multi_face_landmarks[0].landmark])

            # fatigue
            features = landmarks.flatten()
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            probs = torch.softmax(fatigue_model(x), dim=1)
            fatigue_pred = torch.argmax(probs).item()
            fatigue_label = FATIGUE_LABELS[fatigue_pred]

            # blink detection
            left_eye_idx = [33, 160, 158, 133, 153, 144]
            right_eye_idx = [362, 385, 387, 263, 373, 380]
            ear = (eye_aspect_ratio(landmarks, left_eye_idx) + eye_aspect_ratio(landmarks, right_eye_idx)) / 2.0
            if prev_ear > blink_threshold and ear <= blink_threshold:
                blink_count += 1
            prev_ear = ear

            # yawn detection
            mouth_idx = [13, 14]
            mor = mouth_opening_ratio(landmarks, mouth_idx)
            if prev_mor < yawn_threshold and mor >= yawn_threshold:
                yawn_count += 1
            prev_mor = mor

            # head tilt
            tilt_angle = head_tilt_angle(landmarks)

        # emotion (simple)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_crop = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_crop, (48, 48))
            face_tensor = torch.tensor(face_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            probs = torch.softmax(emotion_model(face_tensor), dim=1)
            emotion_pred = torch.argmax(probs).item()
            emotion_label = EMOTION_LABELS[emotion_pred]

        # overlay info
        cv2.putText(frame, f"Fatigue: {fatigue_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        cv2.putText(frame, f"Emotion: {emotion_label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, f"Blinks: {blink_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Yawns: {yawn_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,128,255), 2)
        cv2.putText(frame, f"Head Tilt: {tilt_angle:.1f} deg", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,128,0), 2)

        cv2.imshow("Fatigue + Emotion + Behavior Monitor", frame)

        # log
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data.append([timestamp, fatigue_label, emotion_label, blink_count, yawn_count, tilt_angle])

        if cv2.waitKey(10) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(data, columns=["timestamp", "fatigue", "emotion", "blinks", "yawns", "tilt_angle"])
    df.to_csv(FATIGUE_CSV, index=False)
    print(f"[INFO] Saved {len(df)} records with fatigue, emotion, and activity data to {FATIGUE_CSV}")

# ──────────────────────────────
# RUN
# ──────────────────────────────
if __name__ == "__main__":
    auto_collect_advanced()
