import io
import base64
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Kidney Tumor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

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

ct_model = UNet_CT().to(DEVICE)
try:
    ct_model.load_state_dict(torch.load(MODEL_DIR / "kidney_ct_scan.pth", map_location=DEVICE))
except Exception as e:
    print("Warning: Could not load ct_model:", e)
ct_model.eval()

mri_model = UNet_MRI().to(DEVICE)
try:
    mri_model.load_state_dict(torch.load(MODEL_DIR / "kidney_mri.pth", map_location=DEVICE))
except Exception as e:
    print("Warning: Could not load mri_model:", e)
mri_model.eval()

def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (256, 256))
    img = image / 255.0
    tensor = torch.tensor(img).float().unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE), image

def predict(model, tensor):
    with torch.no_grad():
        pred = model(tensor)
    mask = pred.cpu().numpy()[0][0]
    mask = (mask > 0.5).astype(np.uint8)
    confidence = pred.mean().item() * 100
    return mask, confidence

def compute_measurements(mask):
    tumor_pixels = np.sum(mask)
    if tumor_pixels == 0:
        return 0.0, 0.0
    ys, xs = np.where(mask == 1)
    width = xs.max() - xs.min()
    height = ys.max() - ys.min()
    diameter = max(width, height) * 0.05
    volume = tumor_pixels * 0.0005
    return diameter, volume

def predict_tumor_stage(diameter, tumor_detected):
    if not tumor_detected:
        return "N/A"
    if diameter <= 7.0:
        return "Stage I (T1)"
    else:
        return "Stage II (T2)"

def tumor_localization(mask, image_shape):
    if np.sum(mask) == 0:
        return "No Tumor Detected", None
    ys, xs = np.where(mask == 1)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    center_x = xs.mean()
    center_y = ys.mean()
    height, width = image_shape
    side = "Left Kidney" if center_x < width / 2 else "Right Kidney"
    region = "Upper Cortex Region" if center_y < height / 2 else "Lower Cortex Region"
    bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
    return f"{side} - {region}", bbox

def encode_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/analyze")
async def analyze_scans(ct_file: UploadFile = File(...), mri_file: UploadFile = File(...)):
    ct_data = await ct_file.read()
    mri_data = await mri_file.read()

    ct_img = np.array(Image.open(io.BytesIO(ct_data)).convert("RGB"))
    mri_img = np.array(Image.open(io.BytesIO(mri_data)).convert("RGB"))

    ct_tensor, ct_processed = preprocess(ct_img)
    mri_tensor, mri_processed = preprocess(mri_img)

    ct_mask, ct_conf = predict(ct_model, ct_tensor)
    mri_mask, mri_conf = predict(mri_model, mri_tensor)

    fused_mask = np.logical_or(ct_mask, mri_mask).astype(np.uint8)
    confidence = (ct_conf + mri_conf) / 2
    diameter, volume = compute_measurements(fused_mask)
    localization, bbox = tumor_localization(fused_mask, ct_processed.shape)

    tumor_detected = np.sum(fused_mask) > 0
    diagnosis = "Renal Tumor Detected" if tumor_detected else "No Tumor Detected"
    stage = predict_tumor_stage(diameter, tumor_detected)

    ct_rgb = cv2.cvtColor(ct_processed, cv2.COLOR_GRAY2RGB)
    mri_rgb = cv2.cvtColor(mri_processed, cv2.COLOR_GRAY2RGB)

    ct_mask_display = ct_rgb.copy()
    ct_mask_display[ct_mask == 1] = [255, 69, 0]

    mri_mask_display = mri_rgb.copy()
    mri_mask_display[mri_mask == 1] = [255, 69, 0]

    return JSONResponse({
        "diagnosis": diagnosis,
        "confidence": f"{confidence:.2f}%",
        "volume": f"{volume:.2f} cm³",
        "diameter": f"{diameter:.2f} cm",
        "stage": stage,
        "localization": localization,
        "ctBase64": encode_image(ct_rgb),
        "mriBase64": encode_image(mri_rgb),
        "ctMaskBase64": encode_image(ct_mask_display),
        "mriMaskBase64": encode_image(mri_mask_display)
    })
