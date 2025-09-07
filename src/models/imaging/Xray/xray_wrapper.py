import torch
import numpy as np
from typing import Dict, Any, Optional
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image


class XrayClassifier:
    """
    Wrapper for Hugging Face image classification models for X-Ray tasks
    (e.g., Chest Pneumonia, Limb Fracture).
    """

    def __init__(self, model_id: str, labels: list[str], device: Optional[str] = None):
        self.model_id = model_id
        self.labels = labels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, file_path: str) -> Dict[str, Any]:
        """
        Run inference on an X-ray image file.
        """
        img = Image.open(file_path).convert("RGB")

        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_class = int(np.argmax(probs))
        confidence = float(np.max(probs))
        label = self.labels[pred_class]

        return {
            "label": label,
            "confidence": confidence,
            "heatmap": None, 
            "ai_insight": f"Model suggests **{label}** with {confidence:.2f} confidence."
        }
