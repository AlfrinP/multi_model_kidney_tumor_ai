import time
from fastapi import APIRouter, UploadFile, File, HTTPException, status
import numpy as np
from PIL import Image
import io
import base64

from app.core.preprocess import preprocess
from app.core.model import load_ct_model, load_mri_model, predict
from app.core.postprocess import compute_measurements, predict_tumor_stage, tumor_localization
from app.api.schemas import PredictionResponse, ErrorResponse

api_router = APIRouter()

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    if not filename:
        return False
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_mask_to_base64(mask: np.ndarray) -> str:
    if mask is None:
        return ""
    mask_visual = (mask * 255).astype(np.uint8)
    img = Image.fromarray(mask_visual, mode='L')
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@api_router.get("/health")
def health():
    return {"status": "healthy"}

@api_router.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(ct_image: UploadFile = File(...), mri_image: UploadFile = File(...)):
    start_time = time.time()
    
    if not ct_image.filename or not mri_image.filename:
        raise HTTPException(status_code=400, detail="Empty filename. No file selected.")
    
    if not (allowed_file(ct_image.filename) and allowed_file(mri_image.filename)):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PNG, JPG, JPEG allowed.")
    
    try:
        # Read images
        ct_bytes = await ct_image.read()
        mri_bytes = await mri_image.read()
        ct_img = np.array(Image.open(io.BytesIO(ct_bytes)).convert("RGB"))
        mri_img = np.array(Image.open(io.BytesIO(mri_bytes)).convert("RGB"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image read error: {str(e)}")
    
    # Preprocess
    ct_tensor, ct_gray = preprocess(ct_img)
    mri_tensor, mri_gray = preprocess(mri_img)
    
    # Load models & predict
    ct_model = load_ct_model()
    mri_model = load_mri_model()
    
    ct_mask, ct_conf = predict(ct_model, ct_tensor)
    mri_mask, mri_conf = predict(mri_model, mri_tensor)
    
    # Fusion
    fused_mask = np.logical_or(ct_mask, mri_mask).astype(np.uint8)
    confidence = (ct_conf + mri_conf) / 2.0
    tumor_detected = bool(np.sum(fused_mask) > 0)
    
    # Measurements & staging
    diameter, volume = compute_measurements(fused_mask)
    stage, _ = predict_tumor_stage(diameter, tumor_detected)
    localization, _ = tumor_localization(fused_mask, ct_gray.shape)
    
    additional_info = {
        "tumor_volume_cm3": round(volume, 3),
        "tumor_diameter_cm": round(diameter, 2),
        "predicted_stage": stage,
        "localization": localization,
        "ct_confidence_pct": round(ct_conf, 2),
        "mri_confidence_pct": round(mri_conf, 2)
    }
    
    processing_time = (time.time() - start_time) * 1000.0
    
    return PredictionResponse(
        tumor_detected=tumor_detected,
        cancer_present=tumor_detected,   # same for this model
        probability=round(confidence, 2),
        additional_info=additional_info,
        processing_time_ms=round(processing_time, 2),
        message="Prediction completed successfully",
        ct_mask_base64=encode_mask_to_base64(ct_mask),
        mri_mask_base64=encode_mask_to_base64(mri_mask)
    )
