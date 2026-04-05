"""
MTCNN Face Detector
Detect khuôn mặt trong ảnh sử dụng Multi-task Cascaded CNN
"""

from mtcnn import MTCNN
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple, Dict
from app.core.config import settings
from app.utils.logger import app_logger


class FaceDetector:
    """
    Wrapper class cho MTCNN face detection
    
    MTCNN detect faces và trả về:
    - Bounding box (x, y, width, height)
    - Confidence score
    - Facial landmarks (eyes, nose, mouth corners)
    """
    
    def __init__(self):
        """
        Initialize MTCNN detector với settings từ config
        """
        try:
            # Fix: MTCNN chỉ nhận min_face_size, scale_factor, steps_threshold
            # Không có parameter 'thresholds'
            self.detector = MTCNN(
                min_face_size=settings.MTCNN_MIN_FACE_SIZE,
                steps_threshold=settings.mtcnn_thresholds_list,  # Đổi từ 'thresholds' sang 'steps_threshold'
                scale_factor=settings.MTCNN_FACTOR
            )
            app_logger.info("MTCNN Face Detector initialized successfully")
        except Exception as e:
            app_logger.error(f"Failed to initialize MTCNN: {str(e)}")
            raise
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect tất cả khuôn mặt trong ảnh
        
        Args:
            image: numpy array shape (H, W, 3) RGB
            
        Returns:
            List of dictionaries:
            [
                {
                    'box': [x, y, width, height],
                    'confidence': 0.99,
                    'keypoints': {
                        'left_eye': (x, y),
                        'right_eye': (x, y),
                        'nose': (x, y),
                        'mouth_left': (x, y),
                        'mouth_right': (x, y)
                    }
                },
                ...
            ]
        """
        try:
            results = self.detector.detect_faces(image)
            app_logger.debug(f"Detected {len(results)} face(s)")
            return results
        except Exception as e:
            app_logger.error(f"Face detection failed: {str(e)}")
            return []
    
    def get_largest_face(
        self, 
        image: np.ndarray
    ) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Lấy khuôn mặt lớn nhất trong ảnh
        
        Dùng cho enrollment khi chỉ cần 1 face
        
        Returns:
            (cropped_face, detection_info) hoặc None
        """
        faces = self.detect_faces(image)
        
        if not faces:
            app_logger.warning("No face detected")
            return None
        
        # Tìm face có diện tích lớn nhất
        largest_face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
        
        # Kiểm tra confidence
        if largest_face['confidence'] < settings.MIN_FACE_CONFIDENCE:
            app_logger.warning(
                f"Face confidence {largest_face['confidence']:.2f} "
                f"below threshold {settings.MIN_FACE_CONFIDENCE}"
            )
            return None
        
        # Crop face từ ảnh
        cropped = self._crop_face(image, largest_face['box'])
        
        if cropped is None:
            return None
        
        app_logger.debug(
            f"Largest face: box={largest_face['box']}, "
            f"confidence={largest_face['confidence']:.2f}"
        )
        
        return cropped, largest_face
    
    def _crop_face(
        self, 
        image: np.ndarray, 
        box: List[int],
        margin: float = 0.2
    ) -> Optional[np.ndarray]:
        """
        Crop khuôn mặt từ ảnh với margin
        
        Args:
            image: Ảnh gốc
            box: [x, y, width, height]
            margin: Tỷ lệ margin thêm vào (0.2 = 20%)
            
        Returns:
            Cropped face hoặc None
        """
        try:
            x, y, w, h = box
            
            # Add margin
            margin_x = int(w * margin)
            margin_y = int(h * margin)
            
            # Calculate new coordinates
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(image.shape[1], x + w + margin_x)
            y2 = min(image.shape[0], y + h + margin_y)
            
            # Crop
            cropped = image[y1:y2, x1:x2]
            
            # Validate crop
            if cropped.size == 0:
                app_logger.error("Cropped face is empty")
                return None
            
            return cropped
            
        except Exception as e:
            app_logger.error(f"Face crop failed: {str(e)}")
            return None
    
    def validate_single_face(
        self, 
        image: np.ndarray
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        Validate chỉ có 1 khuôn mặt trong ảnh
        
        Dùng cho cả enrollment và verification
        
        Returns:
            (is_valid, error_message, face_info)
        """
        faces = self.detect_faces(image)
        
        # No face
        if len(faces) == 0:
            return False, "NO_FACE_DETECTED", None
        
        # Multiple faces
        if len(faces) > settings.MAX_FACES_ALLOWED:
            return False, f"MULTIPLE_FACES_DETECTED: {len(faces)} faces found", None
        
        face = faces[0]
        
        # Low confidence
        if face['confidence'] < settings.MIN_FACE_CONFIDENCE:
            return False, f"LOW_CONFIDENCE: {face['confidence']:.2f} < {settings.MIN_FACE_CONFIDENCE}", None
        
        # Check face area ratio
        face_area = face['box'][2] * face['box'][3]
        image_area = image.shape[0] * image.shape[1]
        area_ratio = face_area / image_area
        
        if area_ratio < settings.MIN_FACE_AREA_RATIO:
            return False, f"FACE_TOO_SMALL: area_ratio={area_ratio:.4f} < {settings.MIN_FACE_AREA_RATIO}", None
        
        app_logger.debug(
            f"Valid single face: confidence={face['confidence']:.2f}, "
            f"area_ratio={area_ratio:.4f}"
        )
        
        return True, "VALID", face
    
    def get_face_quality_metrics(self, image: np.ndarray, face_info: Dict) -> Dict:
        """
        Tính các metrics chất lượng khuôn mặt
        
        Returns:
            {
                'confidence': float,
                'size': int,
                'area_ratio': float,
                'symmetry_score': float,
                'brightness': float
            }
        """
        try:
            box = face_info['box']
            x, y, w, h = box
            
            # Face area
            face_area = w * h
            image_area = image.shape[0] * image.shape[1]
            area_ratio = face_area / image_area
            
            # Crop face
            face_crop = image[y:y+h, x:x+w]
            
            # Brightness (mean of grayscale)
            if len(face_crop.shape) == 3:
                gray = np.mean(face_crop, axis=2)
            else:
                gray = face_crop
            brightness = np.mean(gray) / 255.0
            
            # Symmetry (compare left vs right half)
            mid = w // 2
            left_half = face_crop[:, :mid]
            right_half = np.fliplr(face_crop[:, mid:mid + mid])
            
            # Resize to same size
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            # Calculate difference
            diff = np.abs(left_half.astype(float) - right_half.astype(float))
            symmetry_score = 1.0 - (np.mean(diff) / 255.0)
            
            return {
                'confidence': round(face_info['confidence'], 4),
                'size': face_area,
                'area_ratio': round(area_ratio, 4),
                'symmetry_score': round(symmetry_score, 4),
                'brightness': round(brightness, 4)
            }
            
        except Exception as e:
            app_logger.error(f"Failed to calculate face quality metrics: {str(e)}")
            return {
                'confidence': face_info['confidence'],
                'size': 0,
                'area_ratio': 0.0,
                'symmetry_score': 0.0,
                'brightness': 0.0
            }