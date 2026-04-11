import uvicorn
from fastapi import FastAPI
from app.api.routes import api_router
from app.config import Config

app = FastAPI(title="RenalTumorAI API")
app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run("api:app", host=Config.API_HOST, port=Config.API_PORT, reload=False)
