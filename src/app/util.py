
import numpy as np
import pandas as pd
from scipy.signal import resample
from sklearn.preprocessing import scale
import tempfile


EXPECTED_LEN = 256
STEP = 128

PCG_LABELS = [
    "Normal",
    "Aortic Stenosis",
    "Mitral Stenosis",
    "Mitral Valve Prolapse",
    "Pericardial Murmurs"
]

LABELS_EMG = ["healthy", "myopathy", "neuropathy"]

def load_uploaded_file(file, signal_type="ECG") -> np.ndarray:
    name = file.name.lower()

 
    if signal_type in ("ECG", "EMG"):
        text = file.read().decode("utf-8").strip()
        if "," in text:
            vals = [float(x) for x in text.split(",") if x.strip()]
        else:
            vals = [float(x) for x in text.splitlines() if x.strip()]
        return np.array(vals, dtype=np.float32)


def preprocess_signal(x: np.ndarray) -> np.ndarray:
    if x.size != EXPECTED_LEN:
        x = resample(x, EXPECTED_LEN)
    return scale(x).astype(np.float32)   


def segment_signal(raw: np.ndarray) -> np.ndarray:
    raw = preprocess_signal(raw)        
    seg = raw.reshape(EXPECTED_LEN, 1)    
    return seg[np.newaxis, ...]            









def analyze_emg_signal(file, model):
    raw  = load_uploaded_file(file, signal_type="EMG")    
    
    WINDOW = 1000

    wins = []
    if len(raw) < WINDOW:                                
        pad = np.pad(raw, (0, WINDOW - len(raw)))
        wins.append(((pad - pad.mean()) / (pad.std()+1e-6)).reshape(WINDOW, 1))
    else:                                                 
        for i in range(0, len(raw) - WINDOW + 1, WINDOW):
            win = raw[i:i+WINDOW]
            win = (win - win.mean()) / (win.std() + 1e-6)
            wins.append(win.reshape(WINDOW, 1))
    X = np.array(wins, dtype=np.float32)

    preds = model.predict(X, verbose=0)
    classes = np.argmax(preds, axis=1)
    final   = int(np.bincount(classes).argmax())          
    conf    = float(preds[:, final].mean())               
    human   = LABELS_EMG[final]

    return human, conf



