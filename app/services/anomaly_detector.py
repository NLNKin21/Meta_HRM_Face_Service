"""
Anomaly Detection Service
Phát hiện các trường hợp bất thường khi chấm công
"""

from typing import List, Dict
import numpy as np
from app.core.config import settings
from app.utils.logger import app_logger


class AnomalyDetector:
    """
    Detect các anomalies trong quá trình chấm công
    
    Loại anomalies:
    - FACE_MISMATCH: Khuôn mặt không khớp
    - MULTIPLE_FACES: Nhiều khuôn mặt trong ảnh
    - NO_FACE: Không detect được mặt
    - LOW_QUALITY: Chất lượng ảnh thấp
    - LOW_CONFIDENCE: Confidence score thấp
    - SUSPICIOUS_PATTERN: Pattern đáng ngờ
    """
    
    @staticmethod
    def detect_face_anomalies(
        image: np.ndarray,
        face_detection_result: Dict,
        num_faces: int
    ) -> List[Dict]:
        """
        Phát hiện anomalies liên quan đến face detection
        
        Returns:
            List of anomaly dicts
        """
        anomalies = []
        
        # No face detected
        if num_faces == 0:
            anomalies.append({
                'type': 'NO_FACE',
                'severity': 'HIGH',
                'message': 'No face detected in image',
                'metadata': {}
            })
        
        # Multiple faces
        elif num_faces > settings.MAX_FACES_ALLOWED:
            anomalies.append({
                'type': 'MULTIPLE_FACES',
                'severity': 'HIGH',
                'message': f'Detected {num_faces} faces, expected {settings.MAX_FACES_ALLOWED}',
                'metadata': {'num_faces': num_faces}
            })
        
        # Low confidence
        elif face_detection_result:
            confidence = face_detection_result.get('confidence', 0)
            if confidence < settings.MIN_FACE_CONFIDENCE:
                anomalies.append({
                    'type': 'LOW_CONFIDENCE',
                    'severity': 'MEDIUM',
                    'message': f'Face detection confidence {confidence:.2f} below threshold {settings.MIN_FACE_CONFIDENCE}',
                    'metadata': {'confidence': confidence}
                })
        
        return anomalies
    
    @staticmethod
    def detect_image_quality_anomalies(
        quality_score: float,
        image_shape: tuple
    ) -> List[Dict]:
        """
        Phát hiện anomalies về chất lượng ảnh
        """
        anomalies = []
        
        # Low quality
        if quality_score < settings.MIN_IMAGE_QUALITY_SCORE:
            anomalies.append({
                'type': 'LOW_QUALITY',
                'severity': 'MEDIUM',
                'message': f'Image quality score {quality_score:.2f} below threshold {settings.MIN_IMAGE_QUALITY_SCORE}',
                'metadata': {'quality_score': quality_score}
            })
        
        # Image too small
        height, width = image_shape[:2]
        if width < settings.MIN_IMAGE_WIDTH or height < settings.MIN_IMAGE_HEIGHT:
            anomalies.append({
                'type': 'LOW_QUALITY',
                'severity': 'HIGH',
                'message': f'Image size {width}x{height} below minimum {settings.MIN_IMAGE_WIDTH}x{settings.MIN_IMAGE_HEIGHT}',
                'metadata': {'width': width, 'height': height}
            })
        
        return anomalies
    
    @staticmethod
    def detect_verification_anomalies(
        verification_result: Dict,
        match_confidence: float
    ) -> List[Dict]:
        """
        Phát hiện anomalies trong quá trình verification
        """
        anomalies = []
        
        # Face mismatch
        if not verification_result.get('is_match', False):
            anomalies.append({
                'type': 'FACE_MISMATCH',
                'severity': 'CRITICAL',
                'message': f'Face does not match. Confidence: {match_confidence:.2f}%',
                'metadata': {
                    'confidence': match_confidence,
                    'euclidean_distance': verification_result.get('euclidean_distance'),
                    'cosine_similarity': verification_result.get('cosine_similarity')
                }
            })
        
        # Low match confidence (match nhưng confidence thấp)
        elif match_confidence < 80.0:
            anomalies.append({
                'type': 'SUSPICIOUS_PATTERN',
                'severity': 'MEDIUM',
                'message': f'Match found but confidence is low: {match_confidence:.2f}%',
                'metadata': {'confidence': match_confidence}
            })
        
        return anomalies
    
    @staticmethod
    def aggregate_anomalies(anomaly_lists: List[List[Dict]]) -> Dict:
        """
        Tổng hợp tất cả anomalies và tính severity tổng thể
        
        Returns:
            {
                'has_anomalies': bool,
                'total_count': int,
                'severity_level': str,
                'anomalies': List[Dict],
                'summary': str
            }
        """
        # Flatten all anomalies
        all_anomalies = []
        for anomaly_list in anomaly_lists:
            all_anomalies.extend(anomaly_list)
        
        if not all_anomalies:
            return {
                'has_anomalies': False,
                'total_count': 0,
                'severity_level': 'NONE',
                'anomalies': [],
                'summary': 'No anomalies detected'
            }
        
        # Determine overall severity
        severity_priority = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        max_severity = max(
            anomaly['severity'] for anomaly in all_anomalies
        )
        
        # Generate summary
        types = [a['type'] for a in all_anomalies]
        summary = f"Found {len(all_anomalies)} anomaly(ies): {', '.join(set(types))}"
        
        app_logger.warning(f"Anomalies detected: {summary}")
        
        return {
            'has_anomalies': True,
            'total_count': len(all_anomalies),
            'severity_level': max_severity,
            'anomalies': all_anomalies,
            'summary': summary
        }