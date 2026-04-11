import numpy as np
import torch
import torch.nn as nn
from app.config import Config

# ---------- DoubleConv (shared) ----------
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

# ---------- CT U-Net (same as original) ----------
class UNet_CT(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(1, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up3 = DoubleConv(512 + 256, 256)
        self.up2 = DoubleConv(256 + 128, 128)
        self.up1 = DoubleConv(128 + 64, 64)
        self.out = nn.Conv2d(64, 1, 1)
    def forward(self, x):
        c1 = self.down1(x); x = self.pool(c1)
        c2 = self.down2(x); x = self.pool(c2)
        c3 = self.down3(x); x = self.pool(c3)
        x = self.down4(x)
        x = self.upsample(x); x = torch.cat([x, c3], dim=1); x = self.up3(x)
        x = self.upsample(x); x = torch.cat([x, c2], dim=1); x = self.up2(x)
        x = self.upsample(x); x = torch.cat([x, c1], dim=1); x = self.up1(x)
        return torch.sigmoid(self.out(x))

# ---------- MRI U-Net (same as original) ----------
class UNet_MRI(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, 1, 1)
    def forward(self, x):
        d1 = self.down1(x); p1 = self.pool1(d1)
        d2 = self.down2(p1); p2 = self.pool2(d2)
        d3 = self.down3(p2); p3 = self.pool3(d3)
        d4 = self.down4(p3)
        u1 = self.up1(d4); u1 = torch.cat([u1, d3], dim=1); u1 = self.conv1(u1)
        u2 = self.up2(u1); u2 = torch.cat([u2, d2], dim=1); u2 = self.conv2(u2)
        u3 = self.up3(u2); u3 = torch.cat([u3, d1], dim=1); u3 = self.conv3(u3)
        return torch.sigmoid(self.out(u3))

# ---------- Model loader (singleton) ----------
_ct_model = None
_mri_model = None

def load_ct_model():
    global _ct_model
    if _ct_model is None:
        _ct_model = UNet_CT().to(Config.DEVICE)
        _ct_model.load_state_dict(torch.load(Config.CT_MODEL_PATH, map_location=Config.DEVICE))
        _ct_model.eval()
    return _ct_model

def load_mri_model():
    global _mri_model
    if _mri_model is None:
        _mri_model = UNet_MRI().to(Config.DEVICE)
        _mri_model.load_state_dict(torch.load(Config.MRI_MODEL_PATH, map_location=Config.DEVICE))
        _mri_model.eval()
    return _mri_model

def predict(model, tensor):
    with torch.no_grad():
        pred = model(tensor)
    mask = pred.cpu().numpy()[0][0]
    binary_mask = (mask > 0.5).astype(np.uint8)
    tumor_pixels = pred[pred > 0.5]
    if len(tumor_pixels) > 0:
        confidence = tumor_pixels.mean().item() * 100
    else:
        confidence = (1.0 - pred.mean().item()) * 100
    return binary_mask, confidence
