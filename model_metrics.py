import torch
from sklearn.metrics import accuracy_score

def evaluate_model(model, feature_extractor, images, true_labels, label_list):
    preds = []
    for img in images:
        inputs = feature_extractor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        idx = outputs.logits.argmax(-1).item()
        preds.append(label_list[idx])
    acc = accuracy_score(true_labels, preds)
    print(f"Model accuracy: {acc*100:.2f}%")
    return acc

# Example usage:
# images = [Image.open(path) for path in image_paths]
# true_labels = ["Happy", "Sad", ...]
# label_list = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral", "Fatigued"]
# evaluate_model(model, feature_extractor, images, true_labels, label_list)
