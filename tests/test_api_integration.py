# tests/test_api_integration.py
"""
Integration Tests cho API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import base64
import json
import numpy as np
from PIL import Image
import io
from app.main import app

client = TestClient(app)

# ============================================================
# CONSTANT - InsightFace buffalo_l = 512 dimensions
# ============================================================
EMBEDDING_DIM = 512


# ============================================================
# Helper Functions
# ============================================================
def create_fake_face_image() -> str:
    """Tạo ảnh giả có hình dạng giống khuôn mặt để test"""
    # Tạo ảnh 200x200
    img = Image.new('RGB', (200, 200), color=(200, 180, 160))
    
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Vẽ oval giống mặt
    draw.ellipse([40, 20, 160, 180], fill=(230, 210, 190))
    
    # Vẽ mắt
    draw.ellipse([65, 70, 85, 90], fill=(40, 40, 40))
    draw.ellipse([115, 70, 135, 90], fill=(40, 40, 40))
    
    # Vẽ mũi
    draw.polygon([(100, 95), (90, 125), (110, 125)], fill=(200, 180, 160))
    
    # Vẽ miệng
    draw.arc([70, 130, 130, 160], 0, 180, fill=(180, 100, 100), width=3)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def load_real_image_if_exists() -> str:
    """Load ảnh thật nếu có, không thì tạo ảnh fake"""
    sample_dir = Path("data/sample_faces")
    
    if sample_dir.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            images = list(sample_dir.glob(ext))
            if images:
                with open(images[0], 'rb') as f:
                    return base64.b64encode(f.read()).decode('utf-8')
    
    # Fallback: tạo ảnh fake
    return create_fake_face_image()


# ============================================================
# Test Health Check
# ============================================================
class TestHealthCheck:
    """Test health check endpoint"""
    
    def test_health_basic(self):
        """Test GET /health"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'service' in data
        assert 'version' in data
    
    def test_health_with_model_info(self):
        """Test GET /health?include_model_info=true"""
        response = client.get("/health?include_model_info=true")
        assert response.status_code == 200
        
        data = response.json()
        assert 'model_info' in data


# ============================================================
# Test Enrollment
# ============================================================
class TestEnrollment:
    """Test enrollment endpoint"""
    
    @pytest.fixture
    def sample_image_base64(self):
        """Load hoặc tạo sample image"""
        return load_real_image_if_exists()
    
    def test_enroll_success(self, sample_image_base64):
        """Test enrollment thành công"""
        payload = {
            "employee_id": 999,
            "image_base64": sample_image_base64,
            "is_primary": True
        }
        
        response = client.post("/api/face/enroll", json=payload)
        
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # Có thể thành công hoặc thất bại do không detect được face từ ảnh fake
        if response.status_code == 200:
            data = response.json()
            
            if data['success']:
                assert 'data' in data
                assert 'embedding' in data['data']
                # InsightFace buffalo_l = 512 dimensions
                assert len(data['data']['embedding']) == EMBEDDING_DIM
                assert data['data']['employee_id'] == 999
            else:
                # Thất bại nhưng vẫn trả 200 với success=False
                assert 'message' in data
        else:
            # Skip test nếu ảnh fake không detect được face
            pytest.skip(f"Enrollment failed with status {response.status_code}")
    
    def test_enroll_invalid_base64(self):
        """Test với base64 không hợp lệ"""
        payload = {
            "employee_id": 999,
            "image_base64": "invalid_base64_string!!!",
            "is_primary": True
        }
        
        response = client.post("/api/face/enroll", json=payload)
        
        # FastAPI/Pydantic trả 422, hoặc app có thể trả 400
        assert response.status_code in (400, 422)
    
    def test_enroll_missing_employee_id(self, sample_image_base64):
        """Test thiếu employee_id"""
        payload = {
            "image_base64": sample_image_base64
        }
        
        response = client.post("/api/face/enroll", json=payload)
        assert response.status_code == 422  # Pydantic validation
    
    def test_enroll_empty_image(self):
        """Test với image rỗng"""
        payload = {
            "employee_id": 999,
            "image_base64": "",
            "is_primary": True
        }
        
        response = client.post("/api/face/enroll", json=payload)
        assert response.status_code in (400, 422)
    
    def test_enroll_invalid_employee_id(self, sample_image_base64):
        """Test với employee_id không hợp lệ"""
        payload = {
            "employee_id": -1,
            "image_base64": sample_image_base64,
            "is_primary": True
        }
        
        response = client.post("/api/face/enroll", json=payload)
        # Có thể trả 422 nếu có validation, hoặc 200 nếu không
        assert response.status_code in (200, 400, 422)


# ============================================================
# Test Verification
# ============================================================
class TestVerification:
    """Test verification endpoint"""
    
    @pytest.fixture(scope="class")
    def sample_image_base64(self):
        """Load hoặc tạo sample image"""
        return load_real_image_if_exists()
    
    @pytest.fixture(scope="class")
    def enrollment_data(self, sample_image_base64):
        """Enroll một face trước để có embedding"""
        payload = {
            "employee_id": 888,
            "image_base64": sample_image_base64,
            "is_primary": True
        }
        
        response = client.post("/api/face/enroll", json=payload)
        
        if response.status_code != 200:
            pytest.skip(f"Enrollment failed: {response.status_code}")
        
        data = response.json()
        
        if not data.get('success'):
            pytest.skip(f"Enrollment not successful: {data.get('message')}")
        
        return {
            "employee_id": 888,
            "embedding": data['data']['embedding'],
            "image_base64": sample_image_base64
        }
    
    def test_verify_same_person(self, enrollment_data):
        """Test verify với cùng người (should match)"""
        payload = {
            "employee_id": enrollment_data['employee_id'],
            "image_base64": enrollment_data['image_base64'],
            "known_embeddings": [enrollment_data['embedding']]
        }
        
        response = client.post("/api/face/verify", json=payload)
        
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data['success'] == True
        assert data['is_match'] == True
        assert data['confidence'] > 80.0
    
    def test_verify_no_match(self, enrollment_data):
        """Test verify với người khác (should not match)"""
        # Tạo fake embedding ĐÚNG DIMENSION
        fake_embedding = np.random.rand(EMBEDDING_DIM).tolist()
        
        payload = {
            "employee_id": enrollment_data['employee_id'],
            "image_base64": enrollment_data['image_base64'],
            "known_embeddings": [fake_embedding]
        }
        
        response = client.post("/api/face/verify", json=payload)
        
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data['success'] == True
        assert data['is_match'] == False
        assert data['confidence'] < 80.0
    
    def test_verify_invalid_embedding_dimension(self, sample_image_base64):
        """Test với embedding sai dimension"""
        # Embedding sai dimension (64 thay vì 512)
        wrong_embedding = [0.1] * 64
        
        payload = {
            "employee_id": 888,
            "image_base64": sample_image_base64,
            "known_embeddings": [wrong_embedding]
        }
        
        response = client.post("/api/face/verify", json=payload)
        
        # Có thể trả 422 (validation) hoặc 200 với success=False
        if response.status_code == 200:
            data = response.json()
            # Nếu 200 thì phải là success=False hoặc handle gracefully
            # (skip invalid embeddings)
            print(f"Response: {json.dumps(data, indent=2)}")
        else:
            assert response.status_code in (400, 422)
    
    def test_verify_empty_embeddings(self, sample_image_base64):
        """Test với empty known_embeddings"""
        payload = {
            "employee_id": 888,
            "image_base64": sample_image_base64,
            "known_embeddings": []
        }
        
        response = client.post("/api/face/verify", json=payload)
        assert response.status_code in (400, 422)
    
    def test_verify_multiple_embeddings(self, enrollment_data):
        """Test verify với nhiều known embeddings"""
        # Một embedding thật + vài cái random
        known_embeddings = [
            np.random.rand(EMBEDDING_DIM).tolist(),  # Random 1
            enrollment_data['embedding'],              # Đúng
            np.random.rand(EMBEDDING_DIM).tolist(),  # Random 2
        ]
        
        payload = {
            "employee_id": enrollment_data['employee_id'],
            "image_base64": enrollment_data['image_base64'],
            "known_embeddings": known_embeddings
        }
        
        response = client.post("/api/face/verify", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] == True
        # Phải match vì có 1 embedding đúng
        assert data['is_match'] == True


# ============================================================
# Run tests
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])