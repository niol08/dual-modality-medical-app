
from src.models.registry import get_model_for_modality
from src.services.insights import make_insight

def run_inference(modality: str, file_path: str) -> dict:
    """
    Calls the model for the modality and returns a unified result dict:
    { label:str, confidence:float (0..1), ai_insight:str, heatmap:Optional[numpy array or PIL image] }
    """
    model = get_model_for_modality(modality)
    if model is None:
        return {"label":"No model", "confidence":0.0, "ai_insight":"No model registered."}
    out = model.predict(file_path)
    out["ai_insight"] = make_insight(modality, out.get("label"), out.get("confidence"))
    return out
