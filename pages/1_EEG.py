
from utils.page import biosignal_chat_page
from utils.vit_seizure import run_vit_seizure_detector


def run_vit_wrapper(uploaded):
    return run_vit_seizure_detector(
        uploaded,
        repo="JLB-JLB/ViT_Seizure_Detection",
        win_s=6,
        step_s=1,
        image_size=224,
        batch_size=64,
        max_windows=600,  
    )

biosignal_chat_page(
    biosignal_label="EEG",
    slug="eeg",
    accepted_types=("pdf", "txt", "csv", "edf", "fif", "set"),
    analyzer=run_vit_wrapper,                
    analyzer_label="Run EEG Detector" 
)
