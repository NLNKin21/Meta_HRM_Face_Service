"""
Face Enrollment API Routes
"""

from fastapi import APIRouter, HTTPException, status
from app.models.request import FaceEnrollRequest
from app.models.response import FaceEnrollResponse, ErrorResponse
from app.services.enrollment_service import EnrollmentService
from app.utils.logger import app_logger

router = APIRouter(prefix="/api/face", tags=["Face Enrollment"])


@router.post(
    "/enroll",
    response_model=FaceEnrollResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Face enrolled successfully",
            "model": FaceEnrollResponse
        },
        400: {
            "description": "Bad request - Invalid input",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse
        }
    }
)
async def enroll_face(request: FaceEnrollRequest):
    """
    ## Đăng ký khuôn mặt nhân viên mới
    
    Endpoint này nhận ảnh khuôn mặt, xử lý và trả về embedding vector 128 chiều
    để Spring Boot Backend lưu vào database.
    
    ---
    
    ### Workflow:
    
    1. **Decode base64 image**
    2. **Validate image quality**
       - Kích thước ảnh hợp lệ
       - Quality score >= 0.7
    3. **Detect face với MTCNN**
       - Phải có đúng 1 khuôn mặt
       - Face confidence >= 0.95
       - Face area ratio >= 0.05
    4. **Generate embedding với FaceNet**
       - Inception-ResNet-V1
       - Pretrained on VGGFace2
       - Output: 128-dim vector
    5. **Return embedding + metadata**
    
    ---
    
    ### Request Body:
    
    ```json
    {
        "employee_id": 123,
        "image_base64": "iVBORw0KGgo...",
        "is_primary": true,
        "note": "Ảnh đăng ký từ mobile"
    }
    ```
    
    ---
    
    ### Response Success:
    
    ```json
    {
        "success": true,
        "message": "Face enrolled successfully",
        "data": {
            "employee_id": 123,
            "embedding": [0.123, -0.456, ...],  // 128 values
            "face_confidence": 0.99,
            "quality_score": 0.85,
            "face_quality_metrics": {
                "confidence": 0.99,
                "size": 45000,
                "area_ratio": 0.25,
                "symmetry_score": 0.88,
                "brightness": 0.65
            },
            "is_primary": true
        },
        "timestamp": "2024-01-15T10:30:00"
    }
    ```
    
    ---
    
    ### Response Failure (Multiple Faces):
    
    ```json
    {
        "success": false,
        "message": "Face validation failed: MULTIPLE_FACES_DETECTED",
        "data": {
            "anomalies": {
                "has_anomalies": true,
                "total_count": 1,
                "severity_level": "HIGH",
                "anomalies": [
                    {
                        "type": "MULTIPLE_FACES",
                        "severity": "HIGH",
                        "message": "Detected 3 faces, expected 1"
                    }
                ]
            }
        }
    }
    ```
    
    ---
    
    ### Anomaly Detection:
    
    Service tự động phát hiện:
    - ❌ **NO_FACE**: Không có mặt trong ảnh
    - ❌ **MULTIPLE_FACES**: Nhiều hơn 1 mặt
    - ⚠️ **LOW_QUALITY**: Chất lượng ảnh thấp
    - ⚠️ **LOW_CONFIDENCE**: Face detection confidence thấp
    - ⚠️ **FACE_TOO_SMALL**: Khuôn mặt quá nhỏ
    
    ---
    
    ### Integration với Spring Boot:
    
    ```java
    // Java code
    String apiUrl = "http://localhost:8000/api/face/enroll";
    
    FaceEnrollRequest request = new FaceEnrollRequest();
    request.setEmployeeId(employee.getId());
    request.setImageBase64(imageBase64);
    request.setIsPrimary(true);
    
    ResponseEntity<FaceEnrollResponse> response = restTemplate.postForEntity(
        apiUrl,
        request,
        FaceEnrollResponse.class
    );
    
    if (response.getBody().isSuccess()) {
        List<Float> embedding = response.getBody().getData().get("embedding");
        // Lưu embedding vào DB
        employeeFaceService.save(employeeId, embedding);
    }
    ```
    """
    try:
        app_logger.info(
            f"[ENROLL] Request received for employee_id={request.employee_id}"
        )
        
        # Call service
        service = EnrollmentService()
        result = await service.enroll(request)
        
        # Log result
        if result.success:
            app_logger.info(
                f"[ENROLL] Success for employee_id={request.employee_id}"
            )
        else:
            app_logger.warning(
                f"[ENROLL] Failed for employee_id={request.employee_id}: {result.message}"
            )
        
        return result
        
    except ValueError as e:
        # Validation errors
        app_logger.error(f"[ENROLL] Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error": "VALIDATION_ERROR",
                "message": str(e)
            }
        )
    
    except Exception as e:
        # Internal errors
        app_logger.error(f"[ENROLL] Internal error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred"
            }
        )