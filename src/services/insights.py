
def make_insight(modality, label, confidence):
    """
    Small heuristic-based insight generator for the demo.
    Replace with LLM-based summarizer or rule-based clinical insights (with validation).
    """
    conf = float(confidence or 0.0)
    if conf < 0.5:
        return f"Low confidence ({conf:.2f}). Consider re-acquiring higher quality {modality} data or re-running with different preprocessing."
    if "normal" in label.lower() or "no" in label.lower():
        return f"Model suggests '{label}' with confidence {conf:.2f}. This appears benign, but confirm with clinical workflow."
    return f"Model suggests '{label}' with confidence {conf:.2f}. Consider specialist review and correlation with clinical context."
