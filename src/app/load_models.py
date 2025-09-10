
import streamlit as st
import numpy as np
import pandas as pd
from typing import Tuple
import io
from pathlib import Path
from huggingface_hub import hf_hub_download
import tensorflow as tf
from tensorflow import keras
import joblib


from .graph import zeropad, zeropad_output_shape

class HuggingFaceSpaceClient:
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.repo_id = "niol08/Bio-signal-models"
        

        self.models = {
            "ECG": "MLII-latest.keras",
            "PCG": "pcg_model.h5", 
            "EMG": "emg_classifier_txt.h5",
            "VAG": "vag_feature_classifier.pkl"
        }
        
        self.loaded_models = {}

    def _save_uploaded_tmp(self, uploaded_file) -> str:
        suffix = Path(uploaded_file.name).suffix or ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp.flush()
            return tmp.name

    def _load_signal_array(self, path: str) -> np.ndarray:
        """Flexible loader for numeric signal files (CSV, TXT, NPY).
        Returns 1D or 2D numpy array of float32.
        """
        ext = Path(path).suffix.lower()
        if ext == ".npy":
            return np.load(path).astype(np.float32)

        if ext in (".csv", ".txt", ".dat", ""):
            # try to infer separator (comma/whitespace)
            try:
                df = pd.read_csv(path, header=None, comment="#", sep=None, engine="python")
            except Exception:
                df = pd.read_csv(path, header=None, comment="#", delim_whitespace=True)

            # keep numeric columns
            num = df.select_dtypes(include=[np.number])
            if num.empty:
                coerced = df.apply(pd.to_numeric, errors="coerce")
                coerced = coerced.dropna(how="all")
                if coerced.empty:
                    raise Exception("No numeric data found in file")
                arr = coerced.values.astype(np.float32)
            else:
                arr = num.values.astype(np.float32)

            if arr.ndim == 2 and arr.shape[1] == 1:
                return arr[:, 0]
            return arr

        # fallback
        try:
            return np.loadtxt(path).astype(np.float32)
        except Exception:
            raise Exception(f"Unsupported or unreadable file extension: {ext}")
    
    def _download_and_load_model(self, signal_type: str):
        """Download and load model from HuggingFace Hub"""
        if signal_type in self.loaded_models:
            return self.loaded_models[signal_type]
        
        model_filename = self.models[signal_type]
        
        st.info(f"Downloading {model_filename} from HuggingFace...")
        
        try:
            model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=model_filename,
                token=self.hf_token
            )
            
            st.success(f"Downloaded {model_filename}")
            
            if signal_type == "ECG":
                st.info("Loading ECG Keras model with custom functions...")

                model = keras.models.load_model(
                    model_path, 
                    custom_objects={
                        "zeropad": zeropad,
                        "zeropad_output_shape": zeropad_output_shape
                    },
                    compile=False
                )
                
            elif signal_type == "EMG":
                st.info("Loading EMG Keras model...")
                model = keras.models.load_model(model_path, compile=False)
                
            elif signal_type == "VAG":
                st.info("Loading VAG Scikit-learn model...")

                vag_dict = joblib.load(model_path)
                model = vag_dict  

            self.loaded_models[signal_type] = model
            st.success(f"{signal_type} model loaded successfully!")
            
            return model
            
        except Exception as e:
            st.error(f"Failed to download/load {signal_type} model: {str(e)}")
            raise e

    def predict_ecg(self, uploaded_file) -> Tuple[str, str, float]:
        """Predict ECG using MLII-latest.keras from HuggingFace"""
        model = self._download_and_load_model("ECG")

        tmp = self._save_uploaded_tmp(uploaded_file)
        try:
            data = self._load_signal_array(tmp)
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass

        # Convert to 1D signal
        if data.ndim == 2:
            if data.shape[1] >= 1:
                sig = data[:, 0]
            else:
                sig = data.ravel()
        else:
            sig = data

        if sig.size == 0:
            raise Exception("No numeric data found in ECG file")

        # pad/truncate to 256
        if sig.size > 256:
            sig = sig[:256]
        elif sig.size < 256:
            pad = np.zeros(256 - sig.size, dtype=np.float32)
            sig = np.concatenate([sig.astype(np.float32), pad])

        model_input = np.array(sig).reshape(1, 256, 1)

        st.info("Running ECG prediction with HuggingFace model...")

        predictions = model.predict(model_input, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])

        ecg_classes = ["N", "V", "/", "A", "F", "~"]
        class_names = {
            "N": "Normal sinus beat",
            "V": "Premature Ventricular Contraction (PVC)",
            "/": "Paced beat (pacemaker)",
            "A": "Atrial premature beat",
            "F": "Fusion of ventricular & normal beat",
            "~": "Unclassifiable / noise"
        }

        predicted_label = ecg_classes[predicted_class_idx]
        human_readable = class_names.get(predicted_label, "Unknown")

        return predicted_label, human_readable, confidence

    
    def predict_emg(self, uploaded_file) -> Tuple[str, float]:
        """Predict EMG using emg_classifier_txt.h5 from HuggingFace"""
        model = self._download_and_load_model("EMG")

        tmp = self._save_uploaded_tmp(uploaded_file)
        try:
            data = self._load_signal_array(tmp)
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass

        # ensure 1D signal
        if data.ndim == 2:
            # if multi-column, take mean across channels
            if data.shape[1] > 1:
                sig = data.mean(axis=1)
            else:
                sig = data[:, 0]
        else:
            sig = data

        if sig.size == 0:
            raise Exception("No numeric data found in EMG file")

        # pad/truncate to 1000
        if sig.size > 1000:
            sig = sig[:1000]
        elif sig.size < 1000:
            pad = np.zeros(1000 - sig.size, dtype=np.float32)
            sig = np.concatenate([sig.astype(np.float32), pad])

        data_array = sig.astype(np.float32)
        normalized_data = (data_array - data_array.mean()) / (data_array.std() + 1e-6)

        model_input = normalized_data.reshape(1, 1000, 1)

        st.info("Running EMG prediction with HuggingFace model...")

        predictions = model.predict(model_input, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])

        emg_classes = ["healthy", "myopathy", "neuropathy"]
        predicted_label = emg_classes[predicted_class_idx] if predicted_class_idx < len(emg_classes) else "healthy"

        return predicted_label, confidence

    def predict_vag(self, uploaded_file) -> Tuple[str, str, float]:
        """Predict VAG using vag_feature_classifier.pkl from HuggingFace"""
        vag_dict = self._download_and_load_model("VAG")
        
        model = vag_dict["model"]         
        scaler = vag_dict["scaler"]       
        encoder = vag_dict["encoder"]    
        content = uploaded_file.read().decode('utf-8')
        uploaded_file.seek(0)
        
        df = pd.read_csv(io.StringIO(content))
        

        required_features = ['rms_amplitude', 'peak_frequency', 'spectral_entropy', 
                           'zero_crossing_rate', 'mean_frequency']
        
        if not all(feature in df.columns for feature in required_features):
            raise Exception(f"Missing required features. Need: {required_features}")
        
        features = df[required_features].iloc[0].values.reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        st.info("Running VAG prediction with HuggingFace model...")
        
        prediction_encoded = model.predict(features_scaled)[0]
        
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = float(np.max(probabilities))

        prediction_label = encoder.inverse_transform([prediction_encoded])[0]
        human_map = {
            'normal': 'Normal Knee Joint',
            'osteoarthritis': 'Osteoarthritis Detected',
            'ligament_injury': 'Ligament Injury Detected'
        }
        
        human_readable = human_map.get(prediction_label, 'Normal Knee Joint')
        
        return prediction_label.title(), human_readable, confidence