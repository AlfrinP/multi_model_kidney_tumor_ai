# 🩻 Multi-Modal Kidney Tumor AI

A state-of-the-art medical image processing application designed to perform accurate, automated kidney tumor detection, segmentation, and localization using multi-modal AI fusion of CT and MRI scans.

---

## 🚀 Key Features

- **Multi-Modal AI Fusion**: Combines insights from both CT and MRI scan inputs by running them asynchronously through dedicated U-Net models to generate accurate spatial detection.
- **REST API Subsystem**: Decoupled backend powered by FastAPI to serve model inference endpoints.
- **Automated Tumor Segmentation**: Identifies and precisely outlines the shape and bounds of the renal tumor.
- **Volumetric & Spatial Measurements**: Automatically computes critical diagnostic metrics including Tumor Diameter (cm) and Tumor Volume (cm³).
- **Tumor Stage Prediction**: Predicts early versus advanced tumor stages via the TNM clinical categorization system based on physical scan boundaries.
- **Modern Dashboard UI**: Built on an intuitive Streamlit interface, enabling seamless upload, visualization, and analysis of scans.

---

## 🧠 Model Architecture

This project leverages two standalone PyTorch U-Net neural networks tailored for biomedical segmentation:

1. **UNet_CT**: Designed to detect tissue anomalies in high-resolution CT scans.
2. **UNet_MRI**: Optimized for capturing tumor gradients and spatial features in MRI scans.

After independent inference, both segmentation outputs are fused to improve accuracy, enhance confidence, and reduce modality-specific noise.

---

## ⚙️ Installation & Requirements

Ensure you have **Python 3.8+** installed.
Dependencies include: `FastAPI`, `pydantic`, `python-dotenv`, `streamlit`, `torch`, `torchvision`, `opencv-python`, `pillow`, `requests`, `numpy`.

### 🐧 Linux / macOS

```bash
# Navigate to project
cd multi_model_kidney_tumor_ai

# (Optional but recommended) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install "fastapi[standard]" pydantic python-dotenv streamlit torch torchvision opencv-python pillow requests numpy
```

---

### 🪟 Windows (PowerShell)

```powershell
# Navigate to project
cd multi_model_kidney_tumor_ai

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install dependencies
pip install "fastapi[standard]" pydantic python-dotenv streamlit torch torchvision opencv-python pillow requests numpy
```

---

## 💻 Usage Guide

The application is now decoupled into a FastAPI backend and a Streamlit frontend.
First, make sure the `.env` file is properly configured.

1. Start the FastAPI server:
   ```bash
   python api.py
   ```
   (Alternatively, use `fastapi dev api.py` or `uvicorn api:app --reload`)

2. Start the Streamlit application in a separate terminal:
   ```bash
   python streamlit.py
   ```

3. Open your browser and go to:
   ```
   http://localhost:8501
   ```

4. In the sidebar:
   * Enter an optional **Patient Scan ID**
   * Upload corresponding **CT Scan** and **MRI Scan** (`.png`, `.jpg`, `.jpeg`)

5. Click **Run AI Diagnosis**:
   * The streamclient client reaches out to the API backend
   * Performs inference & fusion 
   * Computes tumor metrics
   * Displays diagnostic results
