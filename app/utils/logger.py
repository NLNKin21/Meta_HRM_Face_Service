import sys
from loguru import logger
from app.core.config import settings


def setup_logger():
    """
    Configure logger với file và console output
    """
    
    # Remove default handler
    logger.remove()
    
    # Console handler (màu sắc)
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=settings.LOG_LEVEL
    )
    
    # File handler (plain text)
    logger.add(
        settings.LOG_FILE,
        rotation="10 MB",  # Rotate khi file > 10MB
        retention="30 days",  # Giữ log 30 ngày
        compression="zip",  # Nén file cũ
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level=settings.LOG_LEVEL
    )
    
    return logger


# Global logger instance
app_logger = setup_logger()