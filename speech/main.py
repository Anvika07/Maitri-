import sounddevice as sd
from scipy.io.wavfile import write
from transformers import pipeline
import torch
import os

# -------- SETTINGS --------
SAMPLERATE = 16000
DURATION = 5  # seconds to record from mic
USE_MIC = True   # set False to test with a file (e.g. 'sample.wav')
AUDIO_PATH = "recorded.wav"

# -------- MODEL LOADING --------
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    task="audio-classification",
    model="superb/hubert-base-superb-er",
    device=device
)

# -------- CAPTURE OR LOAD AUDIO --------
if USE_MIC:
    print(f"🎙️ Recording for {DURATION} seconds...")
    audio = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=1)
    sd.wait()
    write(AUDIO_PATH, SAMPLERATE, audio)
    print("✅ Recording saved as", AUDIO_PATH)
else:
    AUDIO_PATH = "sample.wav"

# -------- RUN INFERENCE --------
print("🔍 Analyzing emotion...")
results = classifier(AUDIO_PATH)

print("\n🎧 Detected emotions:")
for r in results:
    print(f"{r['label']:>10s} : {r['score']:.3f}")

# -------- INTERPRET OUTPUT --------
top_emotion = max(results, key=lambda x: x['score'])
print(f"\n🌈 Dominant emotion: {top_emotion['label'].upper()} ({top_emotion['score']:.2f})")
