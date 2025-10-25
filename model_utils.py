<<<<<<< HEAD
from transformers import ViTForImageClassification, ViTFeatureExtractor

def load_model(model_name="mo-thecreator/vit-Facial-Expression-Recognition"):
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    model.eval()  
    return model, feature_extractor
=======
from transformers import ViTForImageClassification, ViTFeatureExtractor

def load_model(model_name="mo-thecreator/vit-Facial-Expression-Recognition"):
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    model.eval()  
    return model, feature_extractor
>>>>>>> 29278773c464dbaaaad8c02d76dc7c4652f8f6fc
