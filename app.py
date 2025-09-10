
import streamlit as st
from src.services.inference import run_inference
from src.utils.io import save_upload_to_tempfile, render_image_preview

st.set_page_config(page_title="DualTech Bot", layout="wide")
st.title("DualTech Bot — Biosignal & Medical Imaging (demo)")

mode = st.sidebar.selectbox("Mode", ["Biosignal", "Medical Imaging"])
if mode == "Biosignal":
    modalities = ["EEG", "ERP", "EOG", "EMG", "ECG"]
else:
    modalities = ["MRI", "CT", "X-ray", "PET", "Angiogram"]

modality = st.sidebar.selectbox("Modality", modalities)
st.sidebar.markdown("Upload a data file supported by the modality (CSV/EDF for biosignals; DICOM/NIfTI/JPEG/PNG for images).")

uploaded_file = st.file_uploader(f"Upload {modality} file", type=["edf","bdf","csv","npy","nii","nii.gz","dcm","dcm.zip","jpg","jpeg","png","dicom","zip","txt"])

if uploaded_file is None:
    st.info("Upload a sample file from `sample_data/` or your device to begin.")
    st.stop()

tmp_path = save_upload_to_tempfile(uploaded_file)

with st.spinner(f"Running {modality} model..."):
    result = run_inference(modality, tmp_path)


st.header("Results")
col1, col2 = st.columns([2,1])
# with col1:
#     st.subheader("Prediction")
#     st.write(f"**Label:** {result.get('label')}")
#     st.write(f"**Confidence:** {result.get('confidence'):.3f}")
#     if result.get("heatmap") is not None:
#         st.subheader("Auxiliary output")
#         st.image(result["heatmap"], caption="Model heatmap / visualization", use_container_width=True)
#     st.subheader("AI Insight")
#     st.info(result.get("ai_insight", "No insight available."))
# inside app.py where you display prediction (replace that block)
with col1:
    st.subheader("Prediction")
    label_name = result.get("label") or result.get("label_name") or "unknown"
    st.write(f"**Label:** {label_name}")
    label_id = result.get("label_id")
    if label_id is None:
        try:
            label_id = int(result.get("label")) if isinstance(result.get("label"), (int, str)) and str(result.get("label")).isdigit() else None
        except Exception:
            label_id = None
    if label_id is not None:
        st.write(f"**Label (id):** {label_id}")
    conf = result.get("confidence") or result.get("score") or 0.0
    try:
        st.write(f"**Confidence:** {conf:.3f}")
    except Exception:
        st.write(f"**Confidence:** {conf}")
    if result.get("heatmap") is not None:
        st.subheader("Auxiliary output")
        st.image(result["heatmap"], caption="Model heatmap / visualization", use_container_width=True)
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
