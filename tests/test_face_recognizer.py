"""
Test FaceNet Face Recognizer
"""
import numpy as np
from PIL import Image
from pathlib import Path
from app.core.face_detector import FaceDetector
from app.core.face_recognizer import FaceRecognizer
from app.utils.distance_calculator import DistanceCalculator
from app.utils.logger import app_logger

def test_get_embedding():
    """Test generate embedding từ cropped face"""
    recognizer = FaceRecognizer()
    
    # Tạo fake face image
    fake_face = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    
    # Get embedding
    embedding = recognizer.get_embedding(fake_face)
    
    assert embedding.shape == (512,)
    print(f"✅ Embedding shape: {embedding.shape}")
    print(f"   Sample values: {embedding[:5]}")

def test_get_embedding_from_real_image():
    """Test pipeline đầy đủ với ảnh thật"""
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    
    test_image_path = Path("data/sample_faces/test_1.jpg")
    if not test_image_path.exists():
        print("⚠️  No test image")
        return
    
    image = np.array(Image.open(test_image_path))
    
    # Get embedding
    embedding = recognizer.get_embedding_from_raw_image(image, detector)
    
    if embedding is not None:
        print(f"✅ Generated embedding: shape={embedding.shape}")
        print(f"   Mean: {np.mean(embedding):.4f}, Std: {np.std(embedding):.4f}")
    else:
        print("❌ Failed to generate embedding")

def test_same_person_recognition():
    """
    Test nhận diện cùng người
    Dùng 1 ảnh, crop 2 lần, so sánh embedding
    """
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    calculator = DistanceCalculator()
    
    test_image_path = Path("data/sample_faces/test_1.jpg")
    if not test_image_path.exists():
        return
    
    image = np.array(Image.open(test_image_path))
    
    # Get embedding 2 lần
    emb1 = recognizer.get_embedding_from_raw_image(image, detector)
    emb2 = recognizer.get_embedding_from_raw_image(image, detector)
    
    if emb1 is not None and emb2 is not None:
        result = calculator.is_same_person(emb1, emb2)
        print(f"✅ Same person test:")
        print(f"   Is match: {result['is_match']}")
        print(f"   Confidence: {result['confidence']:.2f}%")
        print(f"   Euclidean: {result['euclidean_distance']:.4f}")
        print(f"   Cosine: {result['cosine_similarity']:.4f}")

def test_model_info():
    """Test get model info"""
    recognizer = FaceRecognizer()
    info = recognizer.get_model_info()
    
    print("✅ Model Info:")
    for key, value in info.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    test_get_embedding()
    test_get_embedding_from_real_image()
    test_same_person_recognition()
    test_model_info()