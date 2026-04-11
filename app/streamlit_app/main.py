import streamlit as st
import requests
from PIL import Image
import numpy as np
import io
from app.config import Config

st.set_page_config(page_title="RenalTumorAI", page_icon="🩻", layout="wide")

# Custom CSS (same as original, kept for consistency)
st.markdown("""
<style>
.stApp { background-color: #0E1117; }
.css-1d391kg { background-color: #161A22; }
div[data-testid="stMetric"] {
    background-color: #1E2127;
    padding: 15px 20px;
    border-radius: 10px;
    border: 1px solid #2B303B;
}
div[data-testid="stMetric"] label { color: #A0ABC0; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #FFFFFF; }
hr { border-color: #2B303B; }
</style>
""", unsafe_allow_html=True)

st.title("🩻 Kidney Tumor Detection AI")
st.markdown("*Multi‑Modal CT & MRI Fusion for Accurate Renal Tumor Localization*")
st.divider()

API_URL = Config.STREAMLIT_API_URL + "/api/predict"

with st.sidebar:
    st.header("Patient Data Input")
    scan_id = st.text_input("Patient Scan ID", placeholder="e.g. PT-10024")
    st.markdown("### Upload Scans")
    ct_file = st.file_uploader("Upload CT Scan", type=["png", "jpg", "jpeg"])
    mri_file = st.file_uploader("Upload MRI Scan", type=["png", "jpg", "jpeg"])
    st.info("Both CT and MRI are required for fusion analysis.")

if ct_file and mri_file:
    # Display previews
    col1, col2 = st.columns(2)
    with col1:
        st.image(Image.open(ct_file), caption="CT Scan", use_column_width=True)
    with col2:
        st.image(Image.open(mri_file), caption="MRI Scan", use_column_width=True)
    
    if st.button("🔍 Run AI Diagnosis", type="primary"):
        with st.spinner("Contacting AI server..."):
            files = {
                "ct_image": (ct_file.name, ct_file.getvalue(), ct_file.type),
                "mri_image": (mri_file.name, mri_file.getvalue(), mri_file.type)
            }
            try:
                response = requests.post(API_URL, files=files, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    tumor_detected = data["tumor_detected"]
                    diagnosis = "Renal Tumor Detected" if tumor_detected else "No Tumor Detected"
                    diagnosis_color = "🔴" if tumor_detected else "🟢"
                    
                    st.subheader(f"{diagnosis_color} Diagnosis Report: {scan_id if scan_id else 'Unknown Patient'}")
                    
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Diagnosis", diagnosis)
                    m2.metric("Confidence Score", f"{data['probability']:.2f}%")
                    m3.metric("Tumor Volume", f"{data['additional_info']['tumor_volume_cm3']:.2f} cm³")
                    m4.metric("Tumor Diameter", f"{data['additional_info']['tumor_diameter_cm']:.2f} cm")
                    m5.metric("Predicted Stage", data['additional_info']['predicted_stage'])
                    
                    if tumor_detected:
                        st.info(f"**Anatomical Localization:** {data['additional_info']['localization']}")
                    
                    st.divider()
                    st.metric("Processing Time", f"{data['processing_time_ms']:.0f} ms")
                else:
                    st.error(f"API error {response.status_code}: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach API server. Make sure Flask API is running.")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
else:
    st.info("👈 Please upload patient CT and MRI scans in the sidebar.")
