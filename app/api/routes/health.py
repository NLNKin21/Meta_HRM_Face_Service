"""
Health Check Endpoint
Kiểm tra service có hoạt động không
"""

from fastapi import APIRouter, Query
from app.models.response import HealthCheckResponse
from app.core.config import settings
from app.core.face_recognizer import FaceRecognizer
from app.utils.logger import app_logger

router = APIRouter(tags=["Health Check"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    include_model_info: bool = Query(
        default=False,
        description="Có trả về thông tin AI model không"
    )
):
    """
    ## Health Check Endpoint
    
    Kiểm tra service có hoạt động bình thường không.
    
    **Query Parameters:**
    - `include_model_info`: Có trả về thông tin model không (default: false)
    
    **Returns:**
    - `status`: "healthy" hoặc "unhealthy"
    - `service`: Tên service
    - `version`: Version
    - `model_info`: Thông tin model (nếu include_model_info=true)
    
    **Example:**
    ```
    GET /health
    GET /health?include_model_info=true
    ```
    """
    try:
        app_logger.debug("Health check requested")
        
        model_info = None
        
        # If requested, get model info
        if include_model_info:
            try:
                recognizer = FaceRecognizer()
                model_info = recognizer.get_model_info()
            except Exception as e:
                app_logger.error(f"Failed to get model info: {str(e)}")
                model_info = {'error': str(e)}
        
        return HealthCheckResponse(
            status="healthy",
            service=settings.APP_NAME,
            version=settings.APP_VERSION,
            model_info=model_info
        )
        
    except Exception as e:
        app_logger.error(f"Health check failed: {str(e)}")
        return HealthCheckResponse(
            status="unhealthy",
            service=settings.APP_NAME,
            version=settings.APP_VERSION,
            model_info={'error': str(e)}
        )


@router.get("/")
async def root():
    """
    Root endpoint - Service info
    """
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }