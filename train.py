import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import time
from scipy.spatial import distance as dist

# =====================
# Initialize MediaPipe
# =====================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# =====================
# Utility Functions
# =====================
def eye_aspect_ratio(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def get_eye_points(landmarks, indices):
    return np.array([[landmarks[i].x, landmarks[i].y] for i in indices])

# Eye landmarks (MediaPipe)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# =====================
# Parameters
# =====================
EAR_THRESHOLD_SLEEPY = 0.22   # Eye closure threshold
EAR_THRESHOLD_TIRED = 0.26
BLINK_THRESHOLD = 3           # Blinks per 10 sec = tired
DATA_SAVE_INTERVAL = 3        # seconds
DATA_FILE = "fatigue_data.csv"

# =====================
# Initialize
# =====================
cap = cv2.VideoCapture(0)
prev_save_time = time.time()
blink_count = 0
start_time = time.time()

data = []

print("[INFO] Collecting fatigue data automatically... Press ESC to stop.")

# =====================
# Main Loop
# =====================
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    fatigue_label = "Unknown"
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            h, w, _ = frame.shape

            left_eye = get_eye_points(face_landmarks.landmark, LEFT_EYE) * [w, h]
            right_eye = get_eye_points(face_landmarks.landmark, RIGHT_EYE) * [w, h]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Blink detection
            if ear < EAR_THRESHOLD_TIRED:
                blink_count += 1

            # Determine fatigue state
            if ear < EAR_THRESHOLD_SLEEPY:
                fatigue_label = "Sleepy"
                label_val = 2
                color = (0, 0, 255)
            elif ear < EAR_THRESHOLD_TIRED or blink_count > BLINK_THRESHOLD:
                fatigue_label = "Tired"
                label_val = 1
                color = (0, 255, 255)
            else:
                fatigue_label = "Alert"
                label_val = 0
                color = (0, 255, 0)

            # Draw feedback on frame
            cv2.putText(frame, f"Fatigue: {fatigue_label}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"EAR: {ear:.2f}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

            # Save data periodically
            if time.time() - prev_save_time > DATA_SAVE_INTERVAL:
                data.append([ear, blink_count, label_val])
                prev_save_time = time.time()
                print(f"[INFO] Logged: EAR={ear:.3f}, Blinks={blink_count}, Label={fatigue_label}")
                blink_count = 0  # reset for next window

    cv2.imshow("Fatigue Detection - Auto Data Capture", frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
        break

# =====================
# Save Data
# =====================
cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data, columns=["EAR", "Blinks", "Label"])
df.to_csv(DATA_FILE, mode='a', index=False, header=not pd.io.common.file_exists(DATA_FILE))
print(f"[INFO] Data saved to {DATA_FILE}")
