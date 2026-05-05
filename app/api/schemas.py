from pydantic import BaseModel, Field
from typing import Optional

class PredictionResponse(BaseModel):
    tumor_detected: bool
    cancer_present: bool          # same as tumor_detected in this model
    probability: float = Field(..., description="Confidence percentage (0-100)")
    additional_info: dict = Field(..., description="Contains volume, diameter, stage, localization")
    processing_time_ms: float
    message: str = "Prediction completed successfully"
    ct_mask_base64: Optional[str] = Field(None, description="Base64 encoded PNG of the CT mask")
    mri_mask_base64: Optional[str] = Field(None, description="Base64 encoded PNG of the MRI mask")

class ErrorResponse(BaseModel):
    error: str
    message: str
