import uvicorn
from fastapi import FastAPI
from app.api.routes import api_router
from app.config import Config

app = FastAPI(title="RenalTumorAI API")
app.include_router(api_router, prefix="/api")