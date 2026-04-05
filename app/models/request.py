"""
Pydantic Request Models
Validate và serialize request data
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import re


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
        
        # Check if contains base64 characters
        base64_pattern = r'^[A-Za-z0-9+/=]+$'
        
        # If has data URI header, extract base64 part
        if ',' in v:
            v = v.split(',')[1]
        
        # Validate pattern (allow some flexibility)
        if not re.match(base64_pattern, v.replace('\n', '').replace('\r', '')):
            raise ValueError("Invalid base64 format")
        
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
        description="Danh sách các embedding đã lưu của nhân viên"
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
        if ',' in v:
            v = v.split(',')[1]
        
        return v
    
    @field_validator('known_embeddings')
    @classmethod
    def validate_embeddings(cls, v: List[List[float]]) -> List[List[float]]:
        """
        Validate embedding vectors
        - Phải có ít nhất 1 embedding
        - Mỗi embedding phải có 128 chiều
        """
        if not v or len(v) == 0:
            raise ValueError("Must provide at least one known embedding")
        
        for idx, emb in enumerate(v):
            if len(emb) != 128:
                raise ValueError(
                    f"Embedding at index {idx} has {len(emb)} dimensions, expected 128"
                )
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "employee_id": 123,
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                "known_embeddings": [
                    [0.123] * 128,  # Embedding 1
                    [-0.456] * 128  # Embedding 2
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