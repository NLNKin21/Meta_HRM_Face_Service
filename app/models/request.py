"""
Pydantic Request Models
Validate và serialize request data
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import re
import base64


class FaceEnrollRequest(BaseModel):
    """
    Request model cho face enrollment
    
    Example:
    {
        "employee_id": 123,
        "image_base64": "data:image/jpeg;base64,/9j/4AAQ...",
        "is_primary": true
    }
    """
    employee_id: int = Field(
        ..., 
        gt=0,
        description="ID của nhân viên trong hệ thống"
    )
    
    image_base64: str = Field(
        ...,
        min_length=100,
        description="Ảnh khuôn mặt dạng base64 string"
    )
    
    is_primary: bool = Field(
        default=True,
        description="Đánh dấu đây có phải ảnh chính không"
    )
    
    note: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Ghi chú (optional)"
    )
    
    @field_validator('image_base64')
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """
        Validate base64 string
        """
        if not v or len(v) < 100:
            raise ValueError("Image base64 string too short")
        
        # Remove whitespace
        v = v.strip()
        
        # If has data URI header, extract base64 part
        if ',' in v:
            parts = v.split(',')
            if len(parts) != 2:
                raise ValueError("Invalid data URI format")
            v = parts[1]
        
        # Remove whitespace characters
        v_clean = v.replace('\n', '').replace('\r', '').replace(' ', '')
        
        # Validate base64 format
        base64_pattern = r'^[A-Za-z0-9+/]*={0,2}$'
        if not re.match(base64_pattern, v_clean):
            raise ValueError("Invalid base64 format")
        
        # Try to decode to verify
        try:
            decoded = base64.b64decode(v_clean, validate=True)
            if len(decoded) < 100:
                raise ValueError("Decoded image data too small (< 100 bytes)")
        except Exception as e:
            raise ValueError(f"Cannot decode base64: {str(e)}")
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "employee_id": 123,
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                "is_primary": True,
                "note": "Ảnh đăng ký từ mobile app"
            }
        }


class FaceVerifyRequest(BaseModel):
    """
    Request model cho face verification khi chấm công
    
    Example:
    {
        "employee_id": 123,
        "image_base64": "data:image/jpeg;base64,/9j/4AAQ...",
        "known_embeddings": [[0.123, -0.456, ...], [0.234, -0.567, ...]],
        "verification_threshold": 80.0
    }
    """
    employee_id: int = Field(
        ...,
        gt=0,
        description="ID nhân viên cần verify"
    )
    
    image_base64: str = Field(
        ...,
        min_length=100,
        description="Ảnh chấm công dạng base64"
    )
    
    known_embeddings: List[List[float]] = Field(
        ...,
        min_length=1,
        description="Danh sách các embedding đã lưu của nhân viên (512 dimensions)"
    )
    
    verification_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Ngưỡng confidence để coi là match (0-100). Nếu None thì dùng default"
    )
    
    @field_validator('image_base64')
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate base64 image"""
        if not v or len(v) < 100:
            raise ValueError("Image base64 string too short")
        
        v = v.strip()
        
        # Extract base64 from data URI if present
        if ',' in v:
            parts = v.split(',')
            if len(parts) != 2:
                raise ValueError("Invalid data URI format")
            v = parts[1]
        
        # Clean and validate
        v_clean = v.replace('\n', '').replace('\r', '').replace(' ', '')
        
        base64_pattern = r'^[A-Za-z0-9+/]*={0,2}$'
        if not re.match(base64_pattern, v_clean):
            raise ValueError("Invalid base64 format")
        
        try:
            decoded = base64.b64decode(v_clean, validate=True)
            if len(decoded) < 100:
                raise ValueError("Decoded image too small")
        except Exception as e:
            raise ValueError(f"Cannot decode base64: {str(e)}")
        
        return v
    
    @field_validator('known_embeddings')
    @classmethod
    def validate_embeddings(cls, v: List[List[float]]) -> List[List[float]]:
        """
        Validate embedding vectors
        
        ⚠️ KHÔNG validate dimension cứng ở đây vì:
        - Model có thể thay đổi (FaceNet 128-dim, InsightFace 512-dim)
        - Để verification service xử lý dimension mismatch linh hoạt hơn
        
        Chỉ validate:
        - Phải có ít nhất 1 embedding
        - Mỗi embedding không được rỗng
        """
        if not v or len(v) == 0:
            raise ValueError("Must provide at least one known embedding")
        
        for idx, emb in enumerate(v):
            if not emb or len(emb) == 0:
                raise ValueError(f"Embedding at index {idx} is empty")
            
            # Optional: warn if dimension seems unusual (but don't reject)
            if len(emb) not in [128, 512]:
                # Log warning but allow it
                import logging
                logging.getLogger(__name__).warning(
                    f"Unusual embedding dimension at index {idx}: {len(emb)} "
                    f"(expected 128 or 512)"
                )
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "employee_id": 123,
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                "known_embeddings": [
                    [0.123] * 512,  # ✅ Embedding 1 (512-dim for InsightFace)
                    [-0.456] * 512  # ✅ Embedding 2
                ],
                "verification_threshold": 80.0
            }
        }


class HealthCheckRequest(BaseModel):
    """
    Optional request model for health check với test data
    """
    include_model_info: bool = Field(
        default=False,
        description="Có trả về thông tin model không"
    )