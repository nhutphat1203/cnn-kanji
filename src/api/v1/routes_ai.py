from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.responses import JSONResponse
from core.deps import get_reg_kanji_model
from core.logger import setup_logger

logger = setup_logger()
router = APIRouter()

@router.post("/ai/imgreg")
async def get_image_prediction(
    file: UploadFile = File(...),
    model=Depends(get_reg_kanji_model)):
    # Đọc nội dung ảnh
    contents = await file.read()

    # Ở đây bạn có thể xử lý ảnh — ví dụ load model, predict, v.v.
    # result = model.predict(contents)

    # Ví dụ tạm thời: trả về tên file
    result = {"filename": file.filename, "content_type": file.content_type}
    return JSONResponse(content=result)

from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.responses import JSONResponse
from core.deps import get_reg_kanji_model

router = APIRouter()

@router.post("/ai/imgreg")
async def get_image_prediction(
    file: UploadFile = File(...),
    model=Depends(get_reg_kanji_model)
) -> JSONResponse:
    
    try:
        if file.content_type not in ["image/png", "image/jpeg"]:
            return JSONResponse(status_code=400, content={"error": "Invalid file type"})
        contents = await file.read()
        predictions = model.predict(contents)
        result = {"predictions": predictions, "filename": file.filename, "content_type": file.content_type}
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})