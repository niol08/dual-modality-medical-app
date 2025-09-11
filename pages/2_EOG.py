
from utils.page import biosignal_chat_page
from utils.eog_detector import run_eog_detector

biosignal_chat_page(
    biosignal_label="EOG",
    slug="eog",
    accepted_types=("edf", "set", "fif", "txt", "csv"),
    analyzer=lambda uploaded: run_eog_detector(uploaded,
    use_hf_space=False,
    local_model_path=None,  
    win_s=5, step_s=1, max_windows=600),
    analyzer_label="Run EOG detector",
)
