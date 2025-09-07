from pathlib import Path
from src.models.imaging.DeepSA.wrapper import DeepSAWrapper  
from src.models.imaging.CT.swin_wrapper import SwinHFClassifier
from src.models.imaging.PET.wrapper import PETWrapper
from src.models.imaging.Xray.xray_wrapper import XrayClassifier
# from src.models.imaging.MRI.mri_wrapper import MRIClassifier
from src.models.imaging.MRI.mri_wrapper import MRIClassifierHF

from .insights import query_gemini_flash

_mri_service = None
_xray_service = None
_pet_service = None
_swin_service = None
_deepsa_service = None

# def run_inference(modality: str, file_path: str):
#     global _swin_service
#     global _deepsa_service
#     global _pet_service
#     global _xray_service
#     global _mri_service
    
#     if modality == "Angiogram":
#         if _deepsa_service is None:
#             _deepsa_service = DeepSAWrapper(device="cpu")
#         return _deepsa_service.predict(file_path)
#     elif modality.lower() == "ct":
#         if _swin_service is None:
#             _swin_service = SwinHFClassifier(
#                 model_id="Koushim/breast-cancer-swin-classifier",
#                 device="cpu",
#                 hf_token=None
#             )
#         return _swin_service.predict_single(file_path, dicom_windowing=None)
#     elif modality.lower() == "pet":
#         if _pet_service is None:
#             _pet_service = PETWrapper(
#                 model_path="src/models/imaging/PET/models/pet_classifier.h5", 
#                 device="cpu",
#                 target_shape=(128, 128),
#                 suv_threshold=0.5,
#             )
#         return _pet_service.predict(file_path)
#     elif modality.lower() == "x-ray":
#         if _xray_service is None:
#             _xray_service = XrayClassifier(
#                     model_id="prithivMLmods/Bone-Fracture-Detection",
#                     labels=["Fractured", "Not Fractured"],  
#                     device="cpu"
#             )
#         return _xray_service.predict(file_path)
#     elif modality.lower() == "mri":
#         if _mri_service is None:
#             _mri_service = MRIClassifierHF(
#                             model_id="prithivMLmods/BrainTumor-Classification-Mini",
#                             device="cpu"
#                         )
#         return _mri_service.predict(file_path)
#     else:
#         result = {
#             "label": "Not Implemented",
#             "confidence": 0.0,
#             "heatmap": None,           
#             "ai_insight": "No model integrated yet for this modality."
#         }

#     # Add Gemini AI insight if prediction exists
#     if result and "label" in result and "confidence" in result:
#         gemini_insight = query_gemini_flash(modality, result["label"], float(result["confidence"]))
#         result["ai_insight"] = gemini_insight

#     return result

def run_inference(modality: str, file_path: str):
    global _swin_service, _deepsa_service, _pet_service, _xray_service, _mri_service

    if modality == "Angiogram":
        if _deepsa_service is None:
            _deepsa_service = DeepSAWrapper(device="cpu")
        result = _deepsa_service.predict(file_path)
    elif modality.lower() == "ct":
        if _swin_service is None:
            _swin_service = SwinHFClassifier(
                model_id="Koushim/breast-cancer-swin-classifier",
                device="cpu",
                hf_token=None
            )
        result = _swin_service.predict_single(file_path, dicom_windowing=None)
    elif modality.lower() == "pet":
        if _pet_service is None:
            _pet_service = PETWrapper(
                model_path="src/models/imaging/PET/models/pet_classifier.h5", 
                device="cpu",
                target_shape=(128, 128),
                suv_threshold=0.5,
            )
        result = _pet_service.predict(file_path)
    elif modality.lower() == "x-ray":
        if _xray_service is None:
            _xray_service = XrayClassifier(
                model_id="prithivMLmods/Bone-Fracture-Detection",
                labels=["Fractured", "Not Fractured"],  
                device="cpu"
            )
        result = _xray_service.predict(file_path)
    elif modality.lower() == "mri":
        if _mri_service is None:
            _mri_service = MRIClassifierHF(
                model_id="prithivMLmods/BrainTumor-Classification-Mini",
                device="cpu"
            )
        result = _mri_service.predict(file_path)
    else:
        result = {
            "label": "Not Implemented",
            "confidence": 0.0,
            "heatmap": None,           
            "ai_insight": "No model integrated yet for this modality."
        }

    label_name = result.get("label_name") or result.get("label") or "unknown"
    confidence = result.get("confidence") or result.get("score") or 0.0

    result["ai_insight"] = query_gemini_flash(modality, label_name, float(confidence))

    return result
