from transformers import ViTForImageClassification, ViTFeatureExtractor

def load_model(model_name="mo-thecreator/vit-Facial-Expression-Recognition"):
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    model.eval()  
    return model, feature_extractor
