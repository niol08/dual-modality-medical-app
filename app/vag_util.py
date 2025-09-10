

import pandas as pd
import numpy as np

def predict_vag_from_features(file, model, gemini_key=None):
    df = pd.read_csv(file)
    required_features = [
        "rms_amplitude",
        "peak_frequency",
        "spectral_entropy",
        "zero_crossing_rate",
        "mean_frequency"
    ]

    x = df[required_features].values.astype(np.float32)
    preds = model.predict_proba(x)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])

    labels = ["normal", "osteoarthritis", "ligament_injury"]
    label = labels[idx]

    gem_txt = None
    if gemini_key:
        from gemini import query_gemini_rest
        gem_txt = query_gemini_rest("VAG", label, confidence, gemini_key)

    return label, label, confidence, gem_txt
