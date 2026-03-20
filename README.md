# 🩻 Multi-Modal Kidney Tumor AI

A state-of-the-art medical image processing application designed to perform accurate, automated kidney tumor detection, segmentation, and localization using multi-modal AI fusion of CT and MRI scans.

---

## 🚀 Key Features

- **Multi-Modal AI Fusion**: Combines insights from both CT and MRI scan inputs by running them asynchronously through dedicated U-Net models to generate accurate spatial detection.
- **Automated Tumor Segmentation**: Identifies and precisely outlines the shape and bounds of the renal tumor.
- **Volumetric & Spatial Measurements**: Automatically computes critical diagnostic metrics including Tumor Diameter (cm) and Tumor Volume (cm³).
- **Tumor Stage Prediction**: Predicts early versus advanced tumor stages via the TNM clinical categorization system based on physical scan boundaries.
- **Anatomical Localization**: Classifies real-time spatial orientation, automatically determining whether the tumor is on the Left/Right Kidney and Upper/Lower Cortex.
- **Modern Dashboard UI**: Built on an intuitive Streamlit interface, ensuring a seamless experience for medical professionals to upload, view, and analyze scans in real-time.

## 🧠 Model Architecture

This project leverages two standalone PyTorch U-Net neural networks tailored perfectly for biomedical segmentation:
1. **UNet_CT**: A deep U-Net engineered to pinpoint tissue anomalies in high-resolution Computed Tomography (CT) scans.
2. **UNet_MRI**: A corresponding network designed to capture tumor gradients and spatial anomalies via Magnetic Resonance Imaging (MRI).

Upon independent inference, the two segmentation arrays are deterministically fused, boosting final confidence scores while removing noise isolated to individual modalities.

## ⚙️ Installation & Requirements

Ensure you have Python 3.8+ installed. To get started, clone the repository and install the dependencies.

```bash
# Clone the repository
cd multi_model_kidney_tumor_ai

# (Optional but recommended) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

*(Note: If you plan on running the system via the CPU, you will need the CPU version of PyTorch. The inference models support CUDA natively for instantaneous execution.)*

## 💻 Usage Guide

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Navigate to your local host (usually `http://localhost:8501`).
3. On the sidebar, enter an optional **Patient Scan ID**.
4. Upload matching/coregistered **CT Scan** and **MRI Scan** formats (`.png`, `.jpg`, `.jpeg`).
5. **Analyze**: The application instantly runs its forward pass, fuses representations, computes quantitative metrics, and generates a visual diagnostic report complete with processed overlay detections.
