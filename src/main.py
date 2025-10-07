import os
from fastapi import FastAPI
from api.v1.routes_ai import router as ai_router
from web.routes.route_canvas import router as view_router
from core.logger import setup_logger
from fastapi.staticfiles import StaticFiles

logger = setup_logger()

app = FastAPI()

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "web/static")), name="static")

app.include_router(ai_router, prefix="/api/v1")
app.include_router(view_router, prefix="")

@app.get("/health")
async def ping():
    logger.info("Health check endpoint called")
    return {"msg": "ok"}