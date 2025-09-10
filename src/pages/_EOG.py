
from src.utils.eog_detector import run_eog_detector


def run_eog_wrapper(uploaded):
    return run_eog_detector(
        uploaded,
        use_hf_space=False,
        local_model_path=None,
        win_s=5,
        step_s=1,
        max_windows=600,
    )
