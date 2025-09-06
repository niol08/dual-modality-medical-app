
import os
import streamlit as st
from src.services.inference import run_inference
from src.utils.io import save_upload_to_tempfile, render_image_preview

st.set_page_config(page_title="DualTech Bot", layout="wide")
st.title("DualTech Bot — Biosignal & Medical Imaging (demo)")

mode = st.sidebar.selectbox("Mode", ["Biosignal", "Medical Imaging"])
if mode == "Biosignal":
    modalities = ["EEG", "ERP", "EOG", "EMG", "ECG", "MRI (biosignal)"]
else:
    modalities = ["MRI", "CT", "X-ray", "PET", "Angiogram"]

modality = st.sidebar.selectbox("Modality", modalities)
st.sidebar.markdown("Upload a data file supported by the modality (CSV/EDF for biosignals; DICOM/NIfTI/JPEG/PNG for images).")

uploaded_file = st.file_uploader(f"Upload {modality} file", type=["edf","bdf","csv","npy","nii","nii.gz","dcm","dcm.zip","jpg","jpeg","png"])

if uploaded_file is None:
    st.info("Upload a sample file from `sample_data/` or your device to begin.")
    st.stop()

tmp_path = save_upload_to_tempfile(uploaded_file)


def _normalize_result(res: dict) -> dict:
    if not res:
        return {"label": None, "confidence": 0.0}
    r = dict(res)
    # common variants -> label
    if "prediction" in r and "label" not in r:
        r["label"] = r.pop("prediction")
    if "prediction_label" in r and "label" not in r:
        r["label"] = r.pop("prediction_label")
    # confidence synonyms
    if "confidence" not in r:
        for k in ("score", "prob", "probability"):
            if k in r:
                try:
                    r["confidence"] = float(r[k])
                except Exception:
                    r["confidence"] = 0.0
                break
    r.setdefault("label", None)
    r.setdefault("confidence", 0.0)
    return r

if modality == "ERP":
    from src.pages._ERP import run_erp_wrapper
    with st.spinner(f"Running {modality} analyzer..."):
        result = run_erp_wrapper(uploaded_file)
elif modality == "EEG":
    from src.pages._EEG import run_vit_wrapper
    with st.spinner(f"Running {modality} analyzer..."):
        result = run_vit_wrapper(uploaded_file)
elif modality == "EOG":
    from src.pages._EOG import run_eog_wrapper
    with st.spinner(f"Running {modality} analyzer..."):
        result = run_eog_wrapper(uploaded_file)
elif modality == "ECG":
        from src.app.load_models import HuggingFaceSpaceClient
        hf_token = (st.secrets.get("HF_HUB_TOKEN") if hasattr(st, "secrets") else None) or os.getenv("HF_HUB_TOKEN", "")
        client = HuggingFaceSpaceClient(hf_token=hf_token)
        with st.spinner(f"Running {modality} HuggingFace model..."):
            predicted_label, human_readable, confidence = client.predict_ecg(uploaded_file)
        result = {"prediction": predicted_label, "ai_insight": human_readable, "confidence": float(confidence)}
elif modality == "EMG":
        from src.app.load_models import HuggingFaceSpaceClient
        hf_token = (st.secrets.get("HF_HUB_TOKEN") if hasattr(st, "secrets") else None) or os.getenv("HF_HUB_TOKEN", "")
        client = HuggingFaceSpaceClient(hf_token=hf_token)
        with st.spinner(f"Running {modality} HuggingFace model..."):
            predicted_label, confidence = client.predict_emg(uploaded_file)
        result = {"prediction": predicted_label, "confidence": float(confidence)}
else:
    with st.spinner(f"Running {modality} model..."):
        result = run_inference(modality, tmp_path)

result = _normalize_result(result or {})

st.header("Results")
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Prediction")
    st.write(f"**Label:** {result.get('label')}")
    st.write(f"**Confidence:** {result.get('confidence'):.3f}")
    if result.get("heatmap") is not None:
        st.subheader("Auxiliary output")
        st.image(result["heatmap"], caption="Model heatmap / visualization", use_column_width=True)
    st.subheader("AI Insight")
    st.info(result.get("ai_insight", "No insight available."))

with col2:
    st.subheader("Preview")
    try:
        render_image_preview(tmp_path)
    except Exception as e:
        st.text("Preview not available for this file type.")

st.markdown("---")
st.caption("This is a demo scaffold. Replace model stubs in `src/models/` with production models and implement rigorous validation before any clinical use.")
 