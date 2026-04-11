from env import get_env_var

class Config:
    DEVICE = get_env_var("DEVICE", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    CT_MODEL_PATH = get_env_var("CT_MODEL_PATH", default="models/kidney_ct_scan.pth", required=True)
    MRI_MODEL_PATH = get_env_var("MRI_MODEL_PATH", default="models/kidney_mri.pth", required=True)
    API_HOST = get_env_var("API_HOST", default="0.0.0.0")
    API_PORT = int(get_env_var("API_PORT", default="5000"))
    STREAMLIT_API_URL = get_env_var("STREAMLIT_API_URL", default="http://localhost:5000")
