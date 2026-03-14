import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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

st.title("Kidney Tumor Detection (CT + MRI Fusion)")

scan_id = st.text_input("Patient Scan ID")

st.subheader("Upload Scans")

ct_file = st.file_uploader("Upload CT Scan",type=["png","jpg","jpeg"])
mri_file = st.file_uploader("Upload MRI Scan",type=["png","jpg","jpeg"])


if ct_file and mri_file:

    ct_img = np.array(Image.open(ct_file))
    mri_img = np.array(Image.open(mri_file))

    st.image(ct_img,caption="CT Scan")
    st.image(mri_img,caption="MRI Scan")

    ct_tensor,ct_processed = preprocess(ct_img)
    mri_tensor,mri_processed = preprocess(mri_img)

    ct_mask,ct_conf = predict(ct_model,ct_tensor)
    mri_mask,mri_conf = predict(mri_model,mri_tensor)

    # -----------------
    # FUSION
    # -----------------

    fused_mask = np.logical_or(ct_mask,mri_mask).astype(np.uint8)

    confidence = (ct_conf + mri_conf)/2

    diameter,volume = compute_measurements(fused_mask)

    localization,bbox = tumor_localization(fused_mask,ct_processed.shape)

    diagnosis = "Renal Tumor Detected" if np.sum(fused_mask)>0 else "No Tumor Detected"

    overlay = ct_processed.copy()
    overlay = cv2.cvtColor(overlay,cv2.COLOR_GRAY2RGB)

    overlay[fused_mask==1] = [255,0,0]

    if bbox:

        x_min,y_min,x_max,y_max = bbox

        cv2.rectangle(
            overlay,
            (x_min,y_min),
            (x_max,y_max),
            (0,255,0),
            2
        )

    st.subheader("Diagnosis Report")

    st.write("Patient Scan ID:",scan_id)
    st.write("Diagnosis:",diagnosis)
    st.write("Confidence Score:",round(confidence,2),"%")
    st.write("Tumor Diameter:",round(diameter,2),"cm")
    st.write("Tumor Volume:",round(volume,2),"cm³")
    st.write("Tumor Localization:",localization)

    st.subheader("Segmentation Result")

    fig,ax = plt.subplots(1,4,figsize=(16,4))

    ax[0].imshow(ct_processed,cmap="gray")
    ax[0].set_title("CT Scan")

    ax[1].imshow(mri_processed,cmap="gray")
    ax[1].set_title("MRI Scan")

    ax[2].imshow(fused_mask,cmap="hot")
    ax[2].set_title("Fused Mask")

    ax[3].imshow(overlay)
    ax[3].set_title("Overlay")

    for a in ax:
        a.axis("off")

    st.pyplot(fig)