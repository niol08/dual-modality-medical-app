
import streamlit as st
from utils.page import biosignal_chat_page
from utils.egg_detector import predict_egg_full_signal
from utils.egg_loader import load_signal_from_file  

def run_egg_analyzer(uploaded_file):
    sig, fs = load_signal_from_file(uploaded_file)
    return predict_egg_full_signal(sig, fs)

biosignal_chat_page(
    biosignal_label="EGG",
    slug="egg",
    accepted_types=("txt", "csv"),
    analyzer=run_egg_analyzer,
    analyzer_label="Run EGG Analyzer",
)
