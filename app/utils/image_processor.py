"""
Image Processing Utilities
Xử lý ảnh: decode base64, validate, resize, normalize
"""

import base64
import io
import numpy as np
from PIL import Image
from typing import Tuple, Optional
from app.core.config import settings
from app.utils.logger import app_logger


class ImageProcessor:
    """
    Utility class để xử lý ảnh
    """
    
    @staticmethod
    def decode_base64_image(base64_string: str) -> Optional[np.ndarray]:
        """
        Decode base64 string thành numpy array (RGB)
        
        Args:
            base64_string: Base64 encoded image
            
        Returns:
            numpy array shape (H, W, 3) hoặc None nếu lỗi
        """
        try:
            # Remove header nếu có (data:image/jpeg;base64,...)
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode
            image_bytes = base64.b64decode(base64_string)
            
            # Check size
            if len(image_bytes) > settings.MAX_IMAGE_SIZE:
                app_logger.error(f"Image size {len(image_bytes)} exceeds max {settings.MAX_IMAGE_SIZE}")
                return None
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB (nếu là RGBA, grayscale...)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy
            image_array = np.array(image)
            
            app_logger.debug(f"Decoded image shape: {image_array.shape}")
            
            return image_array
            
        except Exception as e:
            app_logger.error(f"Failed to decode base64 image: {str(e)}")
            return None
    
    @staticmethod
    def validate_image_dimensions(image: np.ndarray) -> Tuple[bool, str]:
        """
        Validate kích thước ảnh
        
        Returns:
            (is_valid, error_message)
        """
        height, width = image.shape[:2]
        
        # Check minimum
        if width < settings.MIN_IMAGE_WIDTH or height < settings.MIN_IMAGE_HEIGHT:
            return False, f"Image too small: {width}x{height}. Minimum: {settings.MIN_IMAGE_WIDTH}x{settings.MIN_IMAGE_HEIGHT}"
        
        # Check maximum
        if width > settings.MAX_IMAGE_WIDTH or height > settings.MAX_IMAGE_HEIGHT:
            return False, f"Image too large: {width}x{height}. Maximum: {settings.MAX_IMAGE_WIDTH}x{settings.MAX_IMAGE_HEIGHT}"
        
        return True, "Valid"
    
    @staticmethod
    def calculate_image_quality_score(image: np.ndarray) -> float:
        """
        Tính điểm chất lượng ảnh (0.0 - 1.0)
        
        Dựa trên:
        - Brightness variance
        - Sharpness (Laplacian variance)
        - Contrast
        
        Returns:
            Quality score (0.0 - 1.0)
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image
            
            # 1. Brightness variance (0-1)
            brightness_var = np.var(gray) / (255 ** 2)
            brightness_score = min(brightness_var * 2, 1.0)
            
            # 2. Sharpness (Laplacian variance)
            laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            edges = np.abs(np.convolve(gray.flatten(), laplacian.flatten(), mode='same'))
            sharpness_score = min(np.var(edges) / 10000, 1.0)
            
            # 3. Contrast (standard deviation)
            contrast_score = min(np.std(gray) / 128, 1.0)
            
            # Weighted average
            quality_score = (
                brightness_score * 0.3 +
                sharpness_score * 0.4 +
                contrast_score * 0.3
            )
            
            return float(quality_score)
            
        except Exception as e:
            app_logger.error(f"Failed to calculate quality score: {str(e)}")
            return 0.0
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: int) -> np.ndarray:
        """
        Resize ảnh về target_size x target_size
        
        Args:
            image: numpy array
            target_size: kích thước đích (default 160 cho FaceNet)
            
        Returns:
            Resized image
        """
        pil_image = Image.fromarray(image)
        resized = pil_image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        return np.array(resized)
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values to [-1, 1]
        
        Formula: (pixel - 127.5) / 127.5
        """
        return (image.astype(np.float32) - 127.5) / 127.5