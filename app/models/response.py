"""
Pydantic Response Models
Chuẩn hóa API responses
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class FaceEnrollResponse(BaseModel):
    """
    Response model cho face enrollment
    
    Example Success:
    {
        "success": true,
        "message": "Face enrolled successfully",
        "data": {
            "employee_id": 123,
            "embedding": [0.123, -0.456, ...],  # 512 values
            "face_confidence": 0.99,
            "quality_score": 0.85,
            ...
        }
    }
    """
    success: bool = Field(
        ...,
        description="Request thành công hay không"
    )
    
    message: str = Field(
        ...,
        description="Message mô tả kết quả"
    )
    
    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Data trả về (embedding, metrics, anomalies...)"
    )
    
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(),  # ✅ Fixed
        description="Thời gian xử lý"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Face enrolled successfully",
                "data": {
                    "employee_id": 123,
                    "embedding": [0.123] * 512,  # ✅ 512 dimensions
                    "face_confidence": 0.99,
                    "quality_score": 0.85,
                    "face_quality_metrics": {
                        "confidence": 0.99,
                        "size": 45000,
                        "area_ratio": 0.25,
                        "symmetry_score": 0.88,
                        "brightness": 0.65
                    },
                    "is_primary": True
                },
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class FaceVerifyResponse(BaseModel):
    """
    Response model cho face verification
    """
    success: bool = Field(
        ...,
        description="Request có xử lý thành công không (khác với is_match)"
    )
    
    message: str = Field(
        ...,
        description="Message mô tả kết quả xử lý"
    )
    
    is_match: bool = Field(
        ...,
        description="Khuôn mặt có khớp hay không"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Độ tin cậy của kết quả match (0-100)"
    )
    
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Chi tiết kỹ thuật (distances, similarity scores...)"
    )
    
    anomalies: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Danh sách các anomalies phát hiện được"
    )
    
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(),  # ✅ Fixed
        description="Thời gian xử lý"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Verification completed successfully",
                "is_match": True,
                "confidence": 95.67,
                "details": {
                    "euclidean_distance": 0.45,
                    "cosine_similarity": 0.89,
                    "best_match_index": 0,
                    "method": "BOTH",
                    "num_comparisons": 2
                },
                "anomalies": [],
                "timestamp": "2024-01-15T10:35:00"
            }
        }


class HealthCheckResponse(BaseModel):
    """
    Response cho health check endpoint
    """
    status: str = Field(
        ...,
        description="Status của service: 'healthy' hoặc 'unhealthy'"
    )
    
    service: str = Field(
        ...,
        description="Tên service"
    )
    
    version: str = Field(
        ...,
        description="Version của service"
    )
    
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(),  # ✅ Fixed
        description="Thời gian check"
    )
    
    model_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Thông tin về AI models đang dùng"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "service": "MetaHRM Face Recognition Service",
                "version": "1.0.0",
                "timestamp": "2024-01-15T10:00:00",
                "model_info": {
                    "model_name": "buffalo_l",  # ✅ Updated
                    "provider": "InsightFace",
                    "embedding_dim": 512  # ✅ Updated
                }
            }
        }


class ErrorResponse(BaseModel):
    """
    Standard error response
    """
    success: bool = Field(
        default=False,
        description="Luôn False cho error"
    )
    
    error: str = Field(
        ...,
        description="Error type"
    )
    
    message: str = Field(
        ...,
        description="Error message chi tiết"
    )
    
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Thông tin bổ sung về error"
    )
    
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now()  # ✅ Fixed
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "VALIDATION_ERROR",
                "message": "Image base64 string is invalid",
                "details": {
                    "field": "image_base64",
                    "constraint": "min_length"
                },
                "timestamp": "2024-01-15T10:30:00"
            }
        }