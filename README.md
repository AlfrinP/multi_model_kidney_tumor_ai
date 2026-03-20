# 🩻 Multi-Modal Kidney Tumor AI

A state-of-the-art medical image processing application designed to perform accurate, automated kidney tumor detection, segmentation, and localization using multi-modal AI fusion of CT and MRI scans.

---

## 🚀 Key Features

- **Multi-Modal AI Fusion**: Combines insights from both CT and MRI scan inputs by running them asynchronously through dedicated U-Net models to generate accurate spatial detection.
- **Automated Tumor Segmentation**: Identifies and precisely outlines the shape and bounds of the renal tumor.
- **Volumetric & Spatial Measurements**: Automatically computes critical diagnostic metrics including Tumor Diameter (cm) and Tumor Volume (cm³).
- **Tumor Stage Prediction**: Predicts early versus advanced tumor stages via the TNM clinical categorization system based on physical scan boundaries.
- **Anatomical Localization**: Classifies real-time spatial orientation, automatically determining whether the tumor is on the Left/Right Kidney and Upper/Lower Cortex.
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

### 🐧 Linux / macOS

```bash
# Navigate to project
cd multi_model_kidney_tumor_ai

# (Optional but recommended) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
````

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
pip install -r requirements.txt
```

> ⚠️ If activation fails, run:
>
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

### 🪟 Windows (Command Prompt)

```cmd
:: Navigate to project
cd multi_model_kidney_tumor_ai

:: Create virtual environment
python -m venv .venv

:: Activate virtual environment
.venv\Scripts\activate.bat

:: Install dependencies
pip install -r requirements.txt
```

---

### ⚡ Notes

* Ensure Python is added to your system `PATH`.
* On Linux/macOS, use `python3` if `python` is unavailable.
* For GPU acceleration, install the CUDA-compatible version of PyTorch.
* CPU-only execution is supported but may be slower.

---

## 💻 Usage Guide

1. Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

2. Open your browser and go to:

   ```
   http://localhost:8501
   ```

3. In the sidebar:

   * Enter an optional **Patient Scan ID**
   * Upload corresponding **CT Scan** and **MRI Scan** (`.png`, `.jpg`, `.jpeg`)

4. Click **Analyze**:

   * The system performs inference
   * Fuses CT + MRI outputs
   * Computes tumor metrics
   * Displays segmentation overlays and diagnostic results

---

## 📌 Summary

This system provides an end-to-end pipeline for kidney tumor analysis using multi-modal AI, combining precision, automation, and a user-friendly interface for real-world clinical support.
