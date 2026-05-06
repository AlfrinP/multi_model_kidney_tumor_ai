from env import get_env_var

def _safe_device() -> str:
    import torch
    import torch.nn as nn
    if torch.cuda.is_available():
        try:
            nn.Conv2d(1, 1, 3, padding=1).cuda()(torch.zeros(1, 1, 8, 8).cuda())
            return "cuda"
        except Exception:
            pass
    return "cpu"

class Config:
    DEVICE = get_env_var("DEVICE", default=_safe_device())
    CT_MODEL_PATH = get_env_var("CT_MODEL_PATH", default="models/kidney_ct_scan.pth", required=True)
    MRI_MODEL_PATH = get_env_var("MRI_MODEL_PATH", default="models/kidney_mri.pth", required=True)
    API_HOST = get_env_var("API_HOST", default="0.0.0.0")
    API_PORT = int(get_env_var("API_PORT", default="5000"))
    STREAMLIT_API_URL = get_env_var("STREAMLIT_API_URL", default="http://localhost:5000")
