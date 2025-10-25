import cv2
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt, find_peaks
import mediapipe as mp
import torch
from torch import nn

# -------------------------
# Mediapipe models
# -------------------------
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
mp_pose = mp.solutions.pose.Pose()

# -------------------------
# PyTorch-only emotion recognition
# -------------------------
class SimpleEmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleEmotionCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128,num_classes)
    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

emotion_model = SimpleEmotionCNN()
emotion_model.eval()
emotion_labels = ['angry','disgust','fear','happy','sad','surprise','neutral']

# -------------------------
# Multi-modal LSTM model (dummy)
# -------------------------
class FatigueLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=50, num_classes=3):
        super(FatigueLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

model = FatigueLSTM()
model.eval()

# -------------------------
# Signal processing helpers
# -------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs):
    if len(data)<2: return np.array(data)
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data)

def compute_hr_br(signal, fs):
    if len(signal)<fs*5: return 0,0
    hr_signal = bandpass_filter(signal, 0.7, 4.0, fs)
    peaks,_ = find_peaks(hr_signal, distance=fs*0.3)
    hr_bpm = len(peaks)/(len(hr_signal)/fs)*60
    br_signal = bandpass_filter(signal, 0.1, 0.5, fs)
    br_peaks,_ = find_peaks(br_signal, distance=fs*1.0)
    br_bpm = len(br_peaks)/(len(br_signal)/fs)*60
    return hr_bpm, br_bpm

# -------------------------
# Eye Aspect Ratio (EAR)
# -------------------------
def EAR(eye):
    A = np.linalg.norm(np.array(eye[1])-np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2])-np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0])-np.array(eye[3]))
    return (A+B)/(2.0*C) if C>0 else 0

# -------------------------
# Feature extraction
# -------------------------
def extract_features(frame):
    h, w, _ = frame.shape
    feats = {
        'ear':0.0, 'yawn':0.0, 'slouch':0.0, 'head_tilt':0.0,
        'sleepiness':0.0, 'emotion_happy':0.0
    }

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_res = mp_face_mesh.process(rgb)
    pose_res = mp_pose.process(rgb)

    # Eyes & yawn
    if face_res.multi_face_landmarks:
        lm = face_res.multi_face_landmarks[0]
        pts = [(p.x*w, p.y*h) for p in lm.landmark]
        try:
            left_idx = [33,160,158,133,153,144]
            right_idx = [263,387,385,362,380,373]
            left_eye = [pts[i] for i in left_idx]
            right_eye = [pts[i] for i in right_idx]
            feats['ear'] = (EAR(left_eye)+EAR(right_eye))/2
            feats['yawn'] = np.linalg.norm(np.array(pts[13])-np.array(pts[14]))
            feats['sleepiness'] = 1.0-feats['ear']
        except: pass

    # Head tilt & slouch
    if pose_res.pose_landmarks:
        plm = pose_res.pose_landmarks.landmark
        try:
            sh_mid = ((plm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x +
                       plm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x)/2 * w,
                      (plm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y +
                       plm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y)/2 * h)
            nose = plm[mp.solutions.pose.PoseLandmark.NOSE.value]
            nose_pt = (nose.x*w, nose.y*h)
            feats['head_tilt'] = np.degrees(np.arctan2(nose_pt[1]-sh_mid[1], nose_pt[0]-sh_mid[0]))
            feats['slouch'] = nose_pt[1]-sh_mid[1]
        except: pass

    # Emotion detection via PyTorch CNN
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if face_res.multi_face_landmarks:
            lm = face_res.multi_face_landmarks[0]
            pts = [(p.x*frame.shape[1], p.y*frame.shape[0]) for p in lm.landmark]
            x1 = int(min([p[0] for p in pts]))
            y1 = int(min([p[1] for p in pts]))
            x2 = int(max([p[0] for p in pts]))
            y2 = int(max([p[1] for p in pts]))
            face_crop = gray[y1:y2, x1:x2]
            if face_crop.size>0:
                face_resized = cv2.resize(face_crop,(48,48))
                face_tensor = torch.tensor(face_resized/255.0,dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    outputs = emotion_model(face_tensor)
                    probs = torch.softmax(outputs,dim=1).numpy()[0]
                    feats['emotion_happy'] = float(probs[emotion_labels.index('happy')])
    except: 
        feats['emotion_happy'] = 0.0

    return feats, face_res, pose_res

# -------------------------
# Real-time monitor
# -------------------------
def realtime_monitor():
    cap = cv2.VideoCapture(0)
    buffer_len = 300
    green_signal = deque(maxlen=buffer_len)
    fatigue_history = deque(maxlen=buffer_len)
    hr_history = deque(maxlen=buffer_len)
    br_history = deque(maxlen=buffer_len)
    sleep_history = deque(maxlen=buffer_len)
    emotion_history = deque(maxlen=buffer_len)

    fs = 30
    smoothed_fatigue = 0.0

    while True:
        ret, frame = cap.read()
        if not ret: break

        feats, face_res, pose_res = extract_features(frame)

        # rPPG green channel
        if face_res.multi_face_landmarks:
            lm = face_res.multi_face_landmarks[0]
            pts = [(p.x*frame.shape[1], p.y*frame.shape[0]) for p in lm.landmark]
            forehead_pts = pts[10:16]
            xs = [p[0] for p in forehead_pts]; ys = [p[1] for p in forehead_pts]
            x1, x2 = int(min(xs)), int(max(xs))
            y1, y2 = int(min(ys)), int(max(ys))
            roi = frame[y1:y2, x1:x2, :]
            green_signal.append(np.mean(roi[:,:,1]))

        # Fatigue score
        norm_ear = feats['sleepiness']
        norm_yawn = feats['yawn']/50.0
        norm_slouch = feats['slouch']/50.0
        norm_head = abs(feats['head_tilt'])/30.0
        norm_emotion = 1.0-feats['emotion_happy']
        fatigue_score = 0.35*norm_ear + 0.25*norm_yawn + 0.15*norm_slouch + 0.1*norm_head + 0.15*norm_emotion
        smoothed_fatigue = 0.85*smoothed_fatigue + 0.15*fatigue_score
        fatigue_history.append(smoothed_fatigue)
        sleep_history.append(feats['sleepiness'])
        emotion_history.append(feats['emotion_happy'])

        # Heart rate & breathing
        hr, br = compute_hr_br(list(green_signal), fs)
        hr_history.append(hr)
        br_history.append(br)

        # Status labels
        if smoothed_fatigue<0.3: label='ALERT'
        elif smoothed_fatigue<0.6: label='MILD FATIGUE'
        else: label='FATIGUED'

        # Overlay metrics
        cv2.putText(frame,f'Fatigue: {smoothed_fatigue:.2f} {label}',(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        cv2.putText(frame,f'HR: {hr:.1f} bpm',(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        cv2.putText(frame,f'BR: {br:.1f} bpm',(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)
        cv2.putText(frame,f'Sleepiness: {feats["sleepiness"]:.2f}',(10,120),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
        cv2.putText(frame,f'Emotion(Happy): {feats["emotion_happy"]:.2f}',(10,150),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2)

        # Simple graph overlay
        graph_h, graph_w = 100, 300
        graph = np.zeros((graph_h, graph_w,3),dtype=np.uint8)
        def draw_curve(history, color):
            data = np.array(history)[-graph_w:]
            if len(data)<2: return
            data = np.interp(data,(np.min(data), np.max(data)+1e-5),(graph_h-1,0))
            for i in range(1,len(data)):
                cv2.line(graph,(i-1,int(data[i-1])),(i,int(data[i])),color,1)
        draw_curve(fatigue_history,(0,255,0))
        draw_curve(hr_history,(0,0,255))
        draw_curve(br_history,(255,255,0))
        draw_curve(sleep_history,(0,255,255))
        draw_curve(emotion_history,(255,0,255))
        frame[-graph_h:,-graph_w:] = graph

        cv2.imshow('Wellbeing Monitor + Trends', frame)
        if cv2.waitKey(1)&0xFF==27: break

    cap.release()
    cv2.destroyAllWindows()

# -------------------------
if __name__=="__main__":
    realtime_monitor()
