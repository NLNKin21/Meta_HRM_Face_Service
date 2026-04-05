"""
Face Verification Service
Business logic cho việc verify khuôn mặt khi chấm công
"""

from typing import Dict, List
import numpy as np
from app.core.face_detector import FaceDetector
from app.core.face_recognizer import FaceRecognizer
from app.utils.image_processor import ImageProcessor
from app.utils.distance_calculator import DistanceCalculator
from app.services.anomaly_detector import AnomalyDetector
from app.models.request import FaceVerifyRequest
from app.models.response import FaceVerifyResponse
from app.core.config import settings
from app.utils.logger import app_logger


class VerificationService:
    """
    Service xử lý face verification
    
    Workflow:
    1. Decode base64 image
    2. Validate image quality
    3. Detect face (phải có duy nhất 1 face)
    4. Generate embedding từ face
    5. So sánh với known_embeddings từ DB
    6. Tính confidence score
    7. Phát hiện anomalies
    8. Return kết quả
    """
    
    def __init__(self):
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.distance_calculator = DistanceCalculator()
        self.anomaly_detector = AnomalyDetector()
    
    async def verify(self, request: FaceVerifyRequest) -> FaceVerifyResponse:
        """
        Main verification logic
        
        Args:
            request: FaceVerifyRequest
            
        Returns:
            FaceVerifyResponse
        """
        app_logger.info(
            f"Starting verification for employee_id={request.employee_id}, "
            f"known_embeddings_count={len(request.known_embeddings)}"
        )
        
        try:
            # Step 1: Decode image
            image = ImageProcessor.decode_base64_image(request.image_base64)
            if image is None:
                return FaceVerifyResponse(
                    success=False,
                    message="Failed to decode image",
                    is_match=False,
                    confidence=0.0,
                    details={'error': 'IMAGE_DECODE_FAILED'}
                )
            
            app_logger.debug(f"Image decoded: shape={image.shape}")
            
            # Step 2: Validate dimensions
            is_valid, error_msg = ImageProcessor.validate_image_dimensions(image)
            if not is_valid:
                return FaceVerifyResponse(
                    success=False,
                    message=error_msg,
                    is_match=False,
                    confidence=0.0,
                    details={'error': 'INVALID_DIMENSIONS'}
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
            
            # Step 4: Detect and validate face
            is_valid_face, face_message, face_info = self.face_detector.validate_single_face(image)
            
            if not is_valid_face:
                all_faces = self.face_detector.detect_faces(image)
                face_anomalies = self.anomaly_detector.detect_face_anomalies(
                    image, face_info or {}, len(all_faces)
                )
                all_anomalies.append(face_anomalies)
                
                anomaly_summary = self.anomaly_detector.aggregate_anomalies(all_anomalies)
                
                return FaceVerifyResponse(
                    success=True,  # Request processed nhưng không match
                    message=f"Face validation failed: {face_message}",
                    is_match=False,
                    confidence=0.0,
                    details={
                        'error': 'FACE_VALIDATION_FAILED',
                        'reason': face_message
                    },
                    anomalies=anomaly_summary.get('anomalies', [])
                )
            
            app_logger.debug(f"Face validated: {face_message}")
            
            # Step 5: Extract face and generate embedding
            result = self.face_detector.get_largest_face(image)
            if result is None:
                return FaceVerifyResponse(
                    success=False,
                    message="Failed to extract face",
                    is_match=False,
                    confidence=0.0,
                    details={'error': 'FACE_EXTRACTION_FAILED'}
                )
            
            cropped_face, detection_info = result
            
            # Generate embedding
            query_embedding = self.face_recognizer.get_embedding(cropped_face)
            
            app_logger.debug(f"Generated query embedding: shape={query_embedding.shape}")
            
            # Step 6: Compare with known embeddings
            comparison_result = self._compare_embeddings(
                query_embedding,
                request.known_embeddings,
                request.verification_threshold
            )
            
            # Step 7: Detect verification anomalies
            verification_anomalies = self.anomaly_detector.detect_verification_anomalies(
                comparison_result,
                comparison_result.get('confidence', 0.0)
            )
            all_anomalies.append(verification_anomalies)
            
            # Aggregate all anomalies
            anomaly_summary = self.anomaly_detector.aggregate_anomalies(all_anomalies)
            
            # Step 8: Prepare response
            is_match = comparison_result['is_match']
            confidence = comparison_result['confidence']
            
            # Log result
            app_logger.info(
                f"Verification result for employee_id={request.employee_id}: "
                f"is_match={is_match}, confidence={confidence:.2f}%, "
                f"anomalies={anomaly_summary['total_count']}"
            )
            
            # Build details
            details = {
                'euclidean_distance': comparison_result['best_distance'],
                'cosine_similarity': comparison_result['best_similarity'],
                'best_match_index': comparison_result['best_match_index'],
                'method': comparison_result.get('method', 'UNKNOWN'),
                'num_comparisons': len(request.known_embeddings),
                'face_confidence': face_info['confidence'],
                'quality_score': quality_score,
                'all_comparisons': comparison_result.get('all_comparisons', [])
            }
            
            return FaceVerifyResponse(
                success=True,
                message="Verification completed successfully",
                is_match=is_match,
                confidence=confidence,
                details=details,
                anomalies=anomaly_summary.get('anomalies', []) if anomaly_summary['has_anomalies'] else None
            )
            
        except Exception as e:
            app_logger.error(f"Verification failed: {str(e)}", exc_info=True)
            return FaceVerifyResponse(
                success=False,
                message=f"Internal error: {str(e)}",
                is_match=False,
                confidence=0.0,
                details={'error': 'INTERNAL_ERROR', 'exception': str(e)}
            )
    
    def _compare_embeddings(
        self,
        query_embedding: np.ndarray,
        known_embeddings: List[List[float]],
        custom_threshold: float = None
    ) -> Dict:
        """
        So sánh query embedding với danh sách known embeddings
        
        Args:
            query_embedding: Embedding từ ảnh chấm công
            known_embeddings: List embeddings đã lưu
            custom_threshold: Custom threshold (optional)
            
        Returns:
            Dict chứa kết quả so sánh
        """
        # Convert known embeddings to numpy arrays
        known_embs_np = [np.array(emb) for emb in known_embeddings]
        
        # Find best match
        result = self.distance_calculator.find_best_match(
            query_embedding,
            known_embs_np
        )
        
        # Apply custom threshold if provided
        if custom_threshold is not None:
            # Recalculate is_match based on custom threshold
            if result['confidence'] >= custom_threshold:
                result['is_match'] = True
            else:
                result['is_match'] = False
            
            app_logger.debug(
                f"Applied custom threshold: {custom_threshold}%, "
                f"confidence={result['confidence']:.2f}%, "
                f"is_match={result['is_match']}"
            )
        
        return result