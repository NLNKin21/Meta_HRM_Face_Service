"""
Test MTCNN Face Detector
"""
import numpy as np
from PIL import Image
from pathlib import Path
from app.core.face_detector import FaceDetector
from app.utils.logger import app_logger

def test_detect_faces():
    """Test detect faces"""
    detector = FaceDetector()
    
    # Load test image
    test_image_path = Path("data/sample_faces/test_1.jpg")
    if not test_image_path.exists():
        print("⚠️  No test image found. Run download_test_images.py first")
        return
    
    image = np.array(Image.open(test_image_path))
    
    # Detect
    faces = detector.detect_faces(image)
    
    print(f"✅ Detected {len(faces)} face(s)")
    for idx, face in enumerate(faces):
        print(f"   Face {idx+1}: confidence={face['confidence']:.2f}, box={face['box']}")

def test_get_largest_face():
    """Test get largest face"""
    detector = FaceDetector()
    
    test_image_path = Path("data/sample_faces/test_1.jpg")
    if not test_image_path.exists():
        return
    
    image = np.array(Image.open(test_image_path))
    
    result = detector.get_largest_face(image)
    
    if result:
        cropped, info = result
        print(f"✅ Largest face: {cropped.shape}, confidence={info['confidence']:.2f}")
    else:
        print("❌ No face found")

def test_validate_single_face():
    """Test validate single face"""
    detector = FaceDetector()
    
    test_image_path = Path("data/sample_faces/test_1.jpg")
    if not test_image_path.exists():
        return
    
    image = np.array(Image.open(test_image_path))
    
    is_valid, message, face_info = detector.validate_single_face(image)
    
    print(f"✅ Validation: valid={is_valid}, message={message}")
    if face_info:
        print(f"   Face info: {face_info}")

if __name__ == "__main__":
    test_detect_faces()
    test_get_largest_face()
    test_validate_single_face()