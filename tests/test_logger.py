# File: tests/test_logger.py
from app.utils.logger import app_logger

def test_logger():
    app_logger.info("Test INFO message")
    app_logger.warning("Test WARNING message")
    app_logger.error("Test ERROR message")
    app_logger.success("Test SUCCESS message")
    print("✅ Check logs/face_service.log file")

if __name__ == "__main__":
    test_logger()