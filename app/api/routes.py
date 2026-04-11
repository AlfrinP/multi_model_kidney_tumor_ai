import time
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import io

from app.core.preprocess import preprocess
from app.core.model import load_ct_model, load_mri_model, predict
from app.core.postprocess import compute_measurements, predict_tumor_stage, tumor_localization
from app.api.schemas import PredictionResponse, ErrorResponse

api_bp = Blueprint("api", __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@api_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@api_bp.route("/predict", methods=["POST"])
def predict_endpoint():
    start_time = time.time()
    # Check files
    if "ct_image" not in request.files or "mri_image" not in request.files:
        return jsonify(ErrorResponse(error="Missing files", message="Both ct_image and mri_image are required").dict()), 400
    
    ct_file = request.files["ct_image"]
    mri_file = request.files["mri_image"]
    
    if ct_file.filename == "" or mri_file.filename == "":
        return jsonify(ErrorResponse(error="Empty filename", message="No file selected").dict()), 400
    
    if not (allowed_file(ct_file.filename) and allowed_file(mri_file.filename)):
        return jsonify(ErrorResponse(error="Invalid file type", message="Only PNG, JPG, JPEG allowed").dict()), 400
    
    try:
        # Read images
        ct_img = np.array(Image.open(io.BytesIO(ct_file.read())).convert("RGB"))
        mri_img = np.array(Image.open(io.BytesIO(mri_file.read())).convert("RGB"))
    except Exception as e:
        return jsonify(ErrorResponse(error="Image read error", message=str(e)).dict()), 400
    
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
    
    response = PredictionResponse(
        tumor_detected=tumor_detected,
        cancer_present=tumor_detected,   # same for this model
        probability=round(confidence, 2),
        additional_info=additional_info,
        processing_time_ms=round(processing_time, 2),
        message="Prediction completed successfully"
    )
    
    return jsonify(response.dict())
