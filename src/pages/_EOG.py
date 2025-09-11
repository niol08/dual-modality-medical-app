
def run_eog_wrapper(uploaded):
    try:
        from src.utils.eog_detector import run_eog_detector
    except Exception as e:
        # Import failed (likely missing native dependency like mne/torch). Return a friendly error.
        import streamlit as st
        st.error(f"EOG detector could not be loaded: {e}")
        return {"prediction": None, "confidence": 0.0, "error": str(e)}

    return run_eog_detector(
        uploaded,
        use_hf_space=False,
        local_model_path=None,
        win_s=5,
        step_s=1,
        max_windows=600,
    )
