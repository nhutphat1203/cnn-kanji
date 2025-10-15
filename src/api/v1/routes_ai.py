import base64
import io
import numpy as np
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from core.deps import get_reg_kanji_model, get_model
from core.logger import setup_logger

logger = setup_logger()
router = APIRouter()

# ===== REQUEST/RESPONSE MODELS =====
class PredictRequest(BaseModel):
    image: str  # Base64 encoded image with data URL format

class PredictResponse(BaseModel):
    character: str
    confidence: float
    top5: list[dict]

# ===== HELPER FUNCTIONS =====
def base64_to_image(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    try:
        # Remove data URL prefix if exists
        if ',' in base64_string:
            base64_string = base64_string.split(',', 1)[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_string)
        
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale for model
        if image.mode != 'L':
            image = image.convert('L')
        
        logger.info(f"✅ Decoded image: size={image.size}, mode={image.mode}")
        return image
        
    except Exception as e:
        logger.error(f"❌ Failed to decode base64 image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

def preprocess_image_for_model(image: Image.Image, target_size=(128, 128)) -> np.ndarray:
    """Preprocess PIL Image for CNN model prediction"""
    try:
        # Resize image to model input size
        image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image_resized)
        
        # Normalize pixel values to [0, 1]
        image_array = image_array.astype('float32') / 255.0
        
        # Add channel dimension (height, width, 1)
        image_array = np.expand_dims(image_array, axis=-1)
        
        # Add batch dimension (1, height, width, 1)
        image_array = np.expand_dims(image_array, axis=0)
        
        logger.info(f"✅ Preprocessed image shape: {image_array.shape}")
        return image_array
        
    except Exception as e:
        logger.error(f"❌ Failed to preprocess image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")

# ===== ENDPOINTS =====
@router.post("/predict", response_model=PredictResponse)
async def predict_kanji_from_base64(
    request: PredictRequest,
    model=Depends(get_model)
):
    """Predict kanji character from base64 encoded image (Canvas Drawing)"""
    try:
        logger.info("📥 Received base64 prediction request")
        
        # Step 1: Convert base64 to PIL Image
        image = base64_to_image(request.image)
        
        # Step 2: Preprocess image for model (grayscale 128x128)
        preprocessed_image = preprocess_image_for_model(image, target_size=(128, 128))
        
        # Step 3: Make prediction
        logger.info("🔮 Making prediction...")
        prediction = model.predict(preprocessed_image)
        
        # Step 4: Extract results
        character = prediction['character']
        confidence = float(prediction['confidence'])
        top5 = prediction.get('top5', [])
        
        logger.info(f"✅ Prediction successful: {character} (confidence: {confidence:.4f})")
        
        return PredictResponse(
            character=character,
            confidence=confidence,
            top5=top5
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/imgreg")
async def get_image_prediction(
    file: UploadFile = File(...),
    model=Depends(get_reg_kanji_model)
) -> JSONResponse:
    
    try:
        if file.content_type not in ["image/png", "image/jpeg"]:
            return JSONResponse(status_code=400, content={"error": "Invalid file type"})
        contents = await file.read()
        model.save_pic(file.filename, contents)
        predictions = model.predict(contents)
        result = {"predictions": predictions, "filename": file.filename, "content_type": file.content_type}
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})