
import streamlit as st
import numpy as np
import pandas as pd
from typing import Tuple
import io
from huggingface_hub import hf_hub_download
import tensorflow as tf
from tensorflow import keras
import joblib
import tempfile
import os

from src.app.graph import zeropad, zeropad_output_shape

class HuggingFaceSpaceClient:
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.repo_id = "niol08/Bio-signal-models"
        

        self.models = {
            "ECG": "MLII-latest.keras", 
            "EMG": "emg_classifier_txt.h5",
        }
        
        self.loaded_models = {}
    
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

            self.loaded_models[signal_type] = model
            st.success(f"{signal_type} model loaded successfully!")
            
            return model
            
        except Exception as e:
            st.error(f"Failed to download/load {signal_type} model: {str(e)}")
            raise e

    def predict_ecg(self, uploaded_file) -> Tuple[str, str, float]:
        """Predict ECG using MLII-latest.keras from HuggingFace"""

        model = self._download_and_load_model("ECG")

        content = uploaded_file.read().decode('utf-8')
        uploaded_file.seek(0)
        
        lines = content.strip().split('\n')
        data = []
        for line in lines:
            if ',' in line:
                values = [float(x.strip()) for x in line.split(',') if x.strip()]
                data.extend(values)
            else:
                try:
                    data.append(float(line.strip()))
                except ValueError:
                    continue
        
        if len(data) == 0:
            raise Exception("No numeric data found in ECG file")
        
        if len(data) > 256:
            data = data[:256]
        elif len(data) < 256:
            data.extend([0.0] * (256 - len(data)))
        
        model_input = np.array(data).reshape(1, 256, 1)
        
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
        human_readable = class_names[predicted_label]
        
        return predicted_label, human_readable, confidence

   
        
    def predict_emg(self, uploaded_file) -> Tuple[str, float]:
        """Predict EMG using emg_classifier_txt.h5 from HuggingFace"""
        
        model = self._download_and_load_model("EMG")
        

        content = uploaded_file.read().decode('utf-8')
        uploaded_file.seek(0)
        
        lines = content.strip().split('\n')
        data = []
        for line in lines:
            if ',' in line:
                values = [float(x.strip()) for x in line.split(',') if x.strip()]
                data.extend(values)
            else:
                try:
                    data.append(float(line.strip()))
                except ValueError:
                    continue
        
        if len(data) == 0:
            raise Exception("No numeric data found in EMG file")
    
        if len(data) > 1000:
            data = data[:1000]
        elif len(data) < 1000:
            data.extend([0.0] * (1000 - len(data)))
        

        data_array = np.array(data)
        normalized_data = (data_array - data_array.mean()) / (data_array.std() + 1e-6)
        
        model_input = normalized_data.reshape(1, 1000, 1)
        
        st.info("Running EMG prediction with HuggingFace model...")

        predictions = model.predict(model_input, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])

        emg_classes = ["healthy", "myopathy", "neuropathy"]
        predicted_label = emg_classes[predicted_class_idx] if predicted_class_idx < len(emg_classes) else "healthy"
        
        return predicted_label, confidence

   