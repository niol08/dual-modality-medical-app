
from utils.page import biosignal_chat_page
from utils.erp_hf_model import run_erp_detector

def run_erp_wrapper(uploaded):
    return run_erp_detector(uploaded)

biosignal_chat_page(
    biosignal_label="ERP",
    slug="erp",
    accepted_types=("edf", "fif", "set", "bdf", "csv", "txt"),
    analyzer=run_erp_wrapper,
    analyzer_label="Run ERP Detector"
)
