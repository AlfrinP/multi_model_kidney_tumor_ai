import numpy as np

def compute_measurements(mask: np.ndarray) -> tuple[float, float]:
    tumor_pixels = np.sum(mask)
    if tumor_pixels == 0:
        return 0.0, 0.0
    ys, xs = np.where(mask == 1)
    width = xs.max() - xs.min()
    height = ys.max() - ys.min()
    diameter = max(width, height) * 0.05   # arbitrary scaling
    volume = tumor_pixels * 0.0005
    return diameter, volume

def predict_tumor_stage(diameter: float, tumor_detected: bool) -> tuple[str, str]:
    if not tumor_detected:
        return "N/A", "⚪"
    if diameter <= 7.0:
        return "Stage I (T1)", "🟡"
    else:
        return "Stage II (T2)", "🟠"

def tumor_localization(mask: np.ndarray, image_shape: tuple) -> tuple[str, tuple]:
    if np.sum(mask) == 0:
        return "No Tumor Detected", None
    ys, xs = np.where(mask == 1)
    center_x = xs.mean()
    center_y = ys.mean()
    height, width = image_shape
    side = "Left Kidney" if center_x < width / 2 else "Right Kidney"
    region = "Upper Cortex Region" if center_y < height / 2 else "Lower Cortex Region"
    bbox = (xs.min(), ys.min(), xs.max(), ys.max())
    return f"{side} - {region}", bbox
