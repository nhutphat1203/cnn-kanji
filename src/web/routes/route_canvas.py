# src/web/routes/route.py
import os
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()

# Thư mục template nằm ở web/routes/templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "../templates"))

@router.get("/canvas")
async def canvas_view(request: Request):
    return templates.TemplateResponse("canvas.html", {"request": request})
