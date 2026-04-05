"""
Face Enrollment Service
Business logic cho việc đăng ký khuôn mặt nhân viên
"""

from typing import Dict
import numpy as np
from app.core.face_detector import FaceDetector
from app.core.face_recognizer import FaceRecognizer
from app.utils.image_processor import ImageProcessor
from app.services.anomaly_detector import AnomalyDetector
from app.models.request import FaceEnrollRequest
from app.models.response import FaceEnrollResponse
from app.core.config import settings
from app.utils.logger import app_logger


class EnrollmentService:
    """
    Service xử lý enrollment khuôn mặt
    
    Workflow:
    1. Decode base64 image
    2. Validate image quality
    3. Detect face (phải có duy nhất 1 face)
    4. Generate embedding
    5. Return embedding + metadata
    """
    
    def __init__(self):
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.anomaly_detector = AnomalyDetector()
    
    async def enroll(self, request: FaceEnrollRequest) -> FaceEnrollResponse:
        """
        Main enrollment logic
        
        Args:
            request: FaceEnrollRequest
            
        Returns:
            FaceEnrollResponse
        """
        app_logger.info(f"Starting enrollment for employee_id={request.employee_id}")
        
        try:
            # Step 1: Decode image
            image = ImageProcessor.decode_base64_image(request.image_base64)
            if image is None:
                return FaceEnrollResponse(
                    success=False,
                    message="Failed to decode image",
                    data=None
                )
            
            app_logger.debug(f"Image decoded: shape={image.shape}")
            
            # Step 2: Validate dimensions
            is_valid, error_msg = ImageProcessor.validate_image_dimensions(image)
            if not is_valid:
                return FaceEnrollResponse(
                    success=False,
                    message=error_msg,
                    data=None
                )
            
            # Step 3: Calculate quality score
            quality_score = ImageProcessor.calculate_image_quality_score(image)
            app_logger.debug(f"Image quality score: {quality_score:.2f}")
            
            # Collect anomalies
            all_anomalies = []
            
            # Quality anomalies
            quality_anomalies = self.anomaly_detector.detect_image_quality_anomalies(
                quality_score, image.shape
            )
            all_anomalies.append(quality_anomalies)
            
            # Step 4: Detect face
            is_valid_face, face_message, face_info = self.face_detector.validate_single_face(image)
            
            if not is_valid_face:
                # Count number of faces for better error message
                all_faces = self.face_detector.detect_faces(image)
                face_anomalies = self.anomaly_detector.detect_face_anomalies(
                    image, face_info or {}, len(all_faces)
                )
                all_anomalies.append(face_anomalies)
                
                anomaly_summary = self.anomaly_detector.aggregate_anomalies(all_anomalies)
                
                return FaceEnrollResponse(
                    success=False,
                    message=f"Face validation failed: {face_message}",
                    data={'anomalies': anomaly_summary}
                )
            
            app_logger.debug(f"Face validated: {face_message}")
            
            # Step 5: Get face quality metrics
            face_quality = self.face_detector.get_face_quality_metrics(image, face_info)
            
            # Step 6: Extract and generate embedding
            result = self.face_detector.get_largest_face(image)
            if result is None:
                return FaceEnrollResponse(
                    success=False,
                    message="Failed to extract face",
                    data=None
                )
            
            cropped_face, detection_info = result
            
            # Generate embedding
            embedding = self.face_recognizer.get_embedding(cropped_face)
            
            app_logger.info(
                f"Enrollment successful for employee_id={request.employee_id}, "
                f"embedding_dim={len(embedding)}"
            )
            
            # Prepare response data
            response_data = {
                'employee_id': request.employee_id,
                'embedding': embedding.tolist(),  # Convert numpy to list for JSON
                'face_confidence': face_info['confidence'],
                'quality_score': quality_score,
                'face_quality_metrics': face_quality,
                'is_primary': request.is_primary,
                'detection_info': {
                    'bounding_box': detection_info['box'],
                    'keypoints': detection_info['keypoints']
                }
            }
            
            # Check if there are non-critical anomalies
            anomaly_summary = self.anomaly_detector.aggregate_anomalies(all_anomalies)
            if anomaly_summary['has_anomalies']:
                response_data['anomalies'] = anomaly_summary
            
            return FaceEnrollResponse(
                success=True,
                message="Face enrolled successfully",
                data=response_data
            )
            
        except Exception as e:
            app_logger.error(f"Enrollment failed: {str(e)}", exc_info=True)
            return FaceEnrollResponse(
                success=False,
                message=f"Internal error: {str(e)}",
                data=None
            )