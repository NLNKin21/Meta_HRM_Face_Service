# File: tests/test_image_processor.py
import base64
from PIL import Image
import numpy as np
from app.utils.image_processor import ImageProcessor

def test_decode_base64():
    # Tạo ảnh test
    img = Image.new('RGB', (300, 300), color='red')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_bytes = buffer.getvalue()
    b64_string = base64.b64encode(img_bytes).decode()
    
    # Decode
    decoded = ImageProcessor.decode_base64_image(b64_string)
    assert decoded is not None
    assert decoded.shape == (300, 300, 3)
    print("✅ Base64 decode works!")

def test_validate_dimensions():
    img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    is_valid, msg = ImageProcessor.validate_image_dimensions(img)
    assert is_valid == True
    print("✅ Image validation works!")

def test_quality_score():
    img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    score = ImageProcessor.calculate_image_quality_score(img)
    assert 0.0 <= score <= 1.0
    print(f"✅ Quality score: {score:.2f}")

if __name__ == "__main__":
    import io
    test_decode_base64()
    test_validate_dimensions()
    test_quality_score()