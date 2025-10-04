from fastapi import FastAPI
from api.v1.routes_ai import router as ai_router
from core.logger import setup_logger

logger = setup_logger()

app = FastAPI()
app.include_router(ai_router, prefix="/api/v1")

@app.get("/health")
async def ping():
    logger.info("Health check endpoint called")
    return {"msg": "ok"}