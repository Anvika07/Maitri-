from PIL import Image
import torch

fer_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral", "Fatigued"]
# Expanded mapping to requested emotions
emotion_mapping = {
    "Angry": "Angry",
    "Disgust": "Disgust",
    "Fear": "Stressed",
    "Sad": "Sad",
    "Happy": "Happy",
    "Surprise": "Relaxed",
    "Neutral": "Neutral",
    "Fatigued": "Fatigued"
}

def predict_emotion(model, feature_extractor, image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    idx = outputs.logits.argmax(-1).item()
    raw_emotion = fer_labels[idx]
    mapped_emotion = emotion_mapping.get(raw_emotion, "Neutral")
    return raw_emotion, mapped_emotion
