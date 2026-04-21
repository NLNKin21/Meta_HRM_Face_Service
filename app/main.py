"""
Main FastAPI Application
Entry point cho Face Recognition Service
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from app.core.config import settings
from app.api.routes import enrollment, verification, health
from app.utils.logger import app_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events - Chạy khi startup và shutdown
    """
    # Startup
    app_logger.info("=" * 50)
    app_logger.info(f"Starting {settings.APP_NAME}")
    app_logger.info(f"Version: {settings.APP_VERSION}")
    app_logger.info(f"Debug mode: {settings.DEBUG}")
    app_logger.info(f"FaceNet model: {settings.FACENET_MODEL}")
    app_logger.info("=" * 50)
    
    # Preload models (để warm-up)
    try:
        from app.core.face_detector import FaceDetector
        from app.core.face_recognizer import FaceRecognizer
        
        app_logger.info("Preloading AI models...")
        detector = FaceDetector()
        recognizer = FaceRecognizer()
        app_logger.success("✅ Models loaded successfully")
        
        # Store in app state
        app.state.detector = detector
        app.state.recognizer = recognizer
        
    except Exception as e:
        app_logger.error(f"❌ Failed to load models: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    app_logger.info("Shutting down service...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    ## Face Recognition Service cho hệ thống chấm công MetaHRM
    
    Service này cung cấp 2 API chính:
    
    1. **Face Enrollment** - Đăng ký khuôn mặt nhân viên mới
    2. **Face Verification** - Xác thực khuôn mặt khi chấm công
    
    ### Technology Stack:
    - **Framework**: FastAPI
    - **Face Detection**: MTCNN
    - **Face Recognition**: FaceNet (Inception-ResNet-V1)
    - **Pretrained Model**: VGGFace2
    - **Embedding Dimension**: 512
    
    ### Author:
    - MetaHRM Development Team
    - Khóa luận tốt nghiệp - Hệ thống quản lý nhân sự
    """,
    debug=settings.DEBUG,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


# Include routers
app.include_router(health.router)
app.include_router(enrollment.router)
app.include_router(verification.router)


# Root redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API docs"""
    return RedirectResponse(url="/docs")


# Exception handlers
from fastapi import Request
from fastapi.responses import JSONResponse


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler
    """
    app_logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "detail": str(exc) if settings.DEBUG else None
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )