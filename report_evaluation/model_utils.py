import cv2
import numpy as np
import torch
import torch.nn as nn


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


class UNetCT(nn.Module):
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
        c1 = self.down1(x)
        x = self.pool(c1)
        c2 = self.down2(x)
        x = self.pool(c2)
        c3 = self.down3(x)
        x = self.pool(c3)
        x = self.down4(x)
        x = self.upsample(x)
        x = torch.cat([x, c3], dim=1)
        x = self.up3(x)
        x = self.upsample(x)
        x = torch.cat([x, c2], dim=1)
        x = self.up2(x)
        x = self.upsample(x)
        x = torch.cat([x, c1], dim=1)
        x = self.up1(x)
        return torch.sigmoid(self.out(x))


class UNetMRI(nn.Module):
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
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        u1 = self.up1(d4)
        u1 = torch.cat([u1, d3], dim=1)
        u1 = self.conv1(u1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d1], dim=1)
        u3 = self.conv3(u3)
        return torch.sigmoid(self.out(u3))


def load_models(ct_weights_path="models/kidney_ct_scan.pth", mri_weights_path="models/kidney_mri.pth"):
    ct_model = UNetCT().to(DEVICE)
    ct_model.load_state_dict(torch.load(ct_weights_path, map_location=DEVICE))
    ct_model.eval()

    mri_model = UNetMRI().to(DEVICE)
    mri_model.load_state_dict(torch.load(mri_weights_path, map_location=DEVICE))
    mri_model.eval()
    return ct_model, mri_model


def preprocess_grayscale_image(image: np.ndarray):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (256, 256))
    image_float = image.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_float).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return tensor


def predict_probability_mask(model: nn.Module, image: np.ndarray):
    tensor = preprocess_grayscale_image(image)
    with torch.no_grad():
        pred = model(tensor)
    prob = pred.detach().cpu().numpy()[0, 0]
    return prob
