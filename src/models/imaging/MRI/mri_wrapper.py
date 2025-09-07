
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from typing import Optional, Dict, Any
import numpy as np


class MRIClassifierHF:
    """
    Wrapper for Hugging Face MRI classifier:
    - Model: prithivMLmods/BrainTumor-Classification-Mini
    - Returns human-readable label and confidence
    """

    FALLBACK_LABELS = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]

    def __init__(self, model_id: str = "prithivMLmods/BrainTumor-Classification-Mini", device: Optional[str] = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

 
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

        try:
            raw_labels = list(getattr(self.model.config, "id2label", {}).values())
            if raw_labels and len(raw_labels) == len(self.FALLBACK_LABELS):
                self.labels = raw_labels
            else:
                self.labels = self.FALLBACK_LABELS
        except Exception:
            self.labels = self.FALLBACK_LABELS

        print(f"[MRIClassifierHF] Loaded with labels: {self.labels}")

    def predict(self, file_path: str) -> Dict[str, Any]:
        """
        Run inference on a single MRI image (jpg/png)
        Returns dict: {label, confidence, ai_insight}
        """
        img = Image.open(file_path).convert("RGB")

        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        label = self.labels[pred_idx] if pred_idx < len(self.labels) else f"LABEL_{pred_idx}"

        return {
            "label": label,
            "confidence": confidence,
            "heatmap": None,
            "ai_insight": f"Model suggests **{label}** with {confidence:.2f} confidence."
        }
