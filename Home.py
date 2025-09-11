
import streamlit as st
import sys




st.set_page_config(page_title="Biosignal Chat", page_icon="", layout="wide")


st.title("Biosignal Chat")
st.markdown("""
Welcome! Use the sidebar to pick a biosignal page:

- **EEG** – Electroencephalography  
- **EOG** – Electrooculography  
- **ERP** – Event-Related Potentials  
- **CP** – Carotid pulse as measured by PPG (Photoplethysmography) 
- **EGG** – Electrogastrography  

On each page, upload a relevant file (PDF/CSV/TXT; common biosignal formats like EDF/FIF/SET are also allowed),
then ask questions about the uploaded document or analysis.
""")

st.info("Navigate via the left sidebar. Each page keeps its own chat history and uploaded file.")
st.sidebar.title("Navigation")