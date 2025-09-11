
from utils.page import biosignal_chat_page
from utils.ppg_heartgpt import run_heartgpt_ppg_detector

CKPT = "./HeartGPT/Model_files/PPGPT_500k_iters.zip"  

def run_ppg_wrapper(uploaded):
    return run_heartgpt_ppg_detector(
        uploaded,
        ckpt_path=CKPT,
        heartgpt_dir="./HeartGPT",  
        win_s=10.0,
        step_s=2.0,
        max_windows=600,
    )

biosignal_chat_page(
    biosignal_label="CP (PPG)",
    slug="cp",
    accepted_types=("pdf", "txt", "csv", "edf"),
    analyzer=run_ppg_wrapper,
    analyzer_label="Run CP(PPG) Detector",
)


