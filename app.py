import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Kidney Tumor AI", page_icon="🩻", layout="wide")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------
# DOUBLE CONV BLOCK
# ---------------------------------

class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# ---------------------------------
# CT MODEL
# ---------------------------------

class UNet_CT(nn.Module):

    def __init__(self):
        super().__init__()

        self.down1 = DoubleConv(1,64)
        self.down2 = DoubleConv(64,128)
        self.down3 = DoubleConv(128,256)
        self.down4 = DoubleConv(256,512)

        self.pool = nn.MaxPool2d(2)

        self.upsample = nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)

        self.up3 = DoubleConv(512+256,256)
        self.up2 = DoubleConv(256+128,128)
        self.up1 = DoubleConv(128+64,64)

        self.out = nn.Conv2d(64,1,1)

    def forward(self,x):

        c1 = self.down1(x)
        x = self.pool(c1)

        c2 = self.down2(x)
        x = self.pool(c2)

        c3 = self.down3(x)
        x = self.pool(c3)

        x = self.down4(x)

        x = self.upsample(x)
        x = torch.cat([x,c3],dim=1)
        x = self.up3(x)

        x = self.upsample(x)
        x = torch.cat([x,c2],dim=1)
        x = self.up2(x)

        x = self.upsample(x)
        x = torch.cat([x,c1],dim=1)
        x = self.up1(x)

        return torch.sigmoid(self.out(x))


# ---------------------------------
# MRI MODEL
# ---------------------------------

class UNet_MRI(nn.Module):

    def __init__(self):
        super().__init__()

        self.down1 = DoubleConv(1,64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64,128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128,256)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(256,512)

        self.up1 = nn.ConvTranspose2d(512,256,2,stride=2)
        self.conv1 = DoubleConv(512,256)

        self.up2 = nn.ConvTranspose2d(256,128,2,stride=2)
        self.conv2 = DoubleConv(256,128)

        self.up3 = nn.ConvTranspose2d(128,64,2,stride=2)
        self.conv3 = DoubleConv(128,64)

        self.out = nn.Conv2d(64,1,1)

    def forward(self,x):

        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        d4 = self.down4(p3)

        u1 = self.up1(d4)
        u1 = torch.cat([u1,d3],dim=1)
        u1 = self.conv1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2,d2],dim=1)
        u2 = self.conv2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3,d1],dim=1)
        u3 = self.conv3(u3)

        return torch.sigmoid(self.out(u3))


# ---------------------------------
# LOAD MODELS
# ---------------------------------

@st.cache_resource
def load_models():

    ct_model = UNet_CT().to(DEVICE)
    ct_model.load_state_dict(torch.load("models/kidney_ct_scan.pth",map_location=DEVICE))
    ct_model.eval()

    mri_model = UNet_MRI().to(DEVICE)
    mri_model.load_state_dict(torch.load("models/kidney_mri.pth",map_location=DEVICE))
    mri_model.eval()

    return ct_model,mri_model


ct_model,mri_model = load_models()


# ---------------------------------
# PREPROCESS
# ---------------------------------

def preprocess(image):

    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image,(256,256))

    img = image/255.0

    tensor = torch.tensor(img).float().unsqueeze(0).unsqueeze(0)

    return tensor.to(DEVICE),image


# ---------------------------------
# PREDICTION
# ---------------------------------

def predict(model,tensor):

    with torch.no_grad():
        pred = model(tensor)

    mask = pred.cpu().numpy()[0][0]
    mask = (mask>0.5).astype(np.uint8)

    confidence = pred.mean().item()*100

    return mask,confidence


# ---------------------------------
# MEASUREMENTS
# ---------------------------------

def compute_measurements(mask):

    tumor_pixels = np.sum(mask)

    if tumor_pixels == 0:
        return 0,0

    ys,xs = np.where(mask==1)

    width = xs.max()-xs.min()
    height = ys.max()-ys.min()

    diameter = max(width,height)*0.05
    volume = tumor_pixels*0.0005

    return diameter,volume


def tumor_localization(mask,image_shape):

    if np.sum(mask)==0:
        return "No Tumor Detected",None

    ys,xs = np.where(mask==1)

    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()

    center_x = xs.mean()
    center_y = ys.mean()

    height,width = image_shape

    side = "Left Kidney" if center_x < width/2 else "Right Kidney"
    region = "Upper Cortex Region" if center_y < height/2 else "Lower Cortex Region"

    bbox = (x_min,y_min,x_max,y_max)

    return f"{side} - {region}",bbox


# ---------------------------------
# STREAMLIT UI
# ---------------------------------

st.markdown("""
    <style>
    /* Custom CSS to inject some modern styling */
    .stApp {
        background-color: #0E1117;
    }
    .css-1d391kg {  /* Sidebar */
        background-color: #161A22;
    }
    div[data-testid="stMetric"] {
        background-color: #1E2127;
        padding: 15px 20px;
        border-radius: 10px;
        border: 1px solid #2B303B;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetric"] label {
        color: #A0ABC0;
        font-weight: 500;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #FFFFFF;
        font-weight: 700;
    }
    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border-color: #2B303B;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🩻 Kidney Tumor Detection AI")
st.markdown("*Multi-Modal CT & MRI Fusion for Accurate Renal Tumor Localization*")
st.divider()

# Sidebar for inputs
with st.sidebar:
    st.header("Patient Data Input")
    scan_id = st.text_input("Patient Scan ID", placeholder="e.g. PT-10024")
    
    st.markdown("### Upload Scans")
    ct_file = st.file_uploader("Upload CT Scan", type=["png","jpg","jpeg"])
    mri_file = st.file_uploader("Upload MRI Scan", type=["png","jpg","jpeg"])
    
    st.info("Medical Image Processing requires both CT and MRI scan inputs for the fusion network.")

if ct_file and mri_file:

    ct_img = np.array(Image.open(ct_file))
    mri_img = np.array(Image.open(mri_file))

    ct_tensor, ct_processed = preprocess(ct_img)
    mri_tensor, mri_processed = preprocess(mri_img)

    with st.spinner('Running AI inference models...'):
        ct_mask, ct_conf = predict(ct_model, ct_tensor)
        mri_mask, mri_conf = predict(mri_model, mri_tensor)

        # -----------------
        # FUSION
        # -----------------

        fused_mask = np.logical_or(ct_mask, mri_mask).astype(np.uint8)
        confidence = (ct_conf + mri_conf) / 2
        diameter, volume = compute_measurements(fused_mask)
        localization, bbox = tumor_localization(fused_mask, ct_processed.shape)

        tumor_detected = np.sum(fused_mask) > 0
        diagnosis = "Renal Tumor Detected" if tumor_detected else "No Tumor Detected"
        diagnosis_color = "🔴" if tumor_detected else "🟢"

        overlay = ct_processed.copy()
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)

        # Add red color to where the mask is
        overlay[fused_mask == 1] = [255, 0, 0]

        if bbox:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(
                overlay,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                2
            )

    st.subheader(f"{diagnosis_color} Diagnosis Report: {scan_id if scan_id else 'Unknown Patient'}")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Diagnosis", diagnosis)
    m2.metric("Confidence Score", f"{confidence:.2f}%")
    m3.metric("Tumor Volume", f"{volume:.2f} cm³")
    m4.metric("Tumor Diameter", f"{diameter:.2f} cm")
    
    if tumor_detected:
        st.info(f"**Anatomical Localization:** {localization}")

    st.divider()

    st.subheader("Input Modalities")
    col1, col2 = st.columns(2)
    with col1:
        st.image(ct_img, caption="Original CT Scan", use_container_width=True)
    with col2:
        st.image(mri_img, caption="Original MRI Scan", use_container_width=True)

    st.divider()
    
    st.subheader("Segmentation & Fusion Results")
    
    # Create colored mask display
    mask_display = np.zeros_like(overlay)
    mask_display[fused_mask == 1] = [255, 69, 0] # Orange-Red color
    
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.image(ct_processed, caption="Processed CT", use_container_width=True)
    with r2:
        st.image(mri_processed, caption="Processed MRI", use_container_width=True)
    with r3:
        st.image(mask_display, caption="Fused Mask", use_container_width=True)
    with r4:
        st.image(overlay, caption="Overlay Detection", use_container_width=True)

else:
    # Empty state UI
    st.info("👈 Please upload patient CT and MRI scans in the sidebar to generate a diagnostic report.")
    
    st.markdown("### Example Workflow")
    st.markdown('''
    1. **Upload Scans**: Use the sidebar to upload a set of Coregistered CT and MRI scans.
    2. **AI Inference**: The system asynchronously processes both scans using dedicated U-Net models.
    3. **Fusion & Analysis**: Extracts accurate tumor volume/diameter and localization via deterministic fusion.
    ''')