"""
Configuration Management
Load environment variables và provide settings cho toàn bộ app
"""

from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application Settings từ .env file
    Sử dụng Pydantic để validate và type checking
    """
    
    # ============================================
    # APPLICATION
    # ============================================
    APP_NAME: str
    APP_VERSION: str
    DEBUG: bool
    HOST: str
    PORT: int
    
    # ============================================
    # MODEL CONFIGURATION
    # ============================================
    FACENET_MODEL: str
    MTCNN_MIN_FACE_SIZE: int
    MTCNN_THRESHOLDS: str
    MTCNN_FACTOR: float
    
    # ============================================
    # RECOGNITION THRESHOLDS
    # ============================================
    EUCLIDEAN_THRESHOLD: float
    COSINE_THRESHOLD: float
    MIN_FACE_CONFIDENCE: float
    
    # ============================================
    # IMAGE PROCESSING
    # ============================================
    MAX_IMAGE_SIZE: int
    ALLOWED_EXTENSIONS: str
    TARGET_FACE_SIZE: int
    MIN_IMAGE_WIDTH: int
    MIN_IMAGE_HEIGHT: int
    MAX_IMAGE_WIDTH: int
    MAX_IMAGE_HEIGHT: int
    
    # ============================================
    # ANOMALY DETECTION
    # ============================================
    MIN_IMAGE_QUALITY_SCORE: float
    MAX_FACES_ALLOWED: int
    MIN_FACE_AREA_RATIO: float
    
    # ============================================
    # CORS
    # ============================================
    ALLOWED_ORIGINS: str
    
    # ============================================
    # LOGGING
    # ============================================
    LOG_LEVEL: str
    LOG_FILE: str
    
    # ============================================
    # COMPUTED PROPERTIES
    # ============================================
    
    @property
    def allowed_origins_list(self) -> List[str]:
        """Convert comma-separated origins to list"""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(',')]
    
    @property
    def mtcnn_thresholds_list(self) -> List[float]:
        """Convert comma-separated thresholds to list of floats"""
        return [float(x.strip()) for x in self.MTCNN_THRESHOLDS.split(',')]
    
    @property
    def allowed_extensions_list(self) -> List[str]:
        """Convert comma-separated extensions to list"""
        return [ext.strip().lower() for ext in self.ALLOWED_EXTENSIONS.split(',')]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Cache settings instance để tránh load lại nhiều lần
    """
    return Settings()


# Global settings instance
settings = get_settings()