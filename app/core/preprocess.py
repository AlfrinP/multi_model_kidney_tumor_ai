import cv2
import torch
import numpy as np
from app.config import Config

def preprocess(image: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    """
    Convert RGB image to grayscale, resize to 256x256, normalize,
    and return tensor (1,1,256,256) + grayscale image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (256, 256))
    normalized = resized / 255.0
    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    try:
        return tensor.to(Config.DEVICE), resized
    except RuntimeError:
        Config.DEVICE = "cpu"
        return tensor.cpu(), resized
