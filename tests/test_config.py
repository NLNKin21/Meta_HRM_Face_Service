# File: tests/test_config.py
from app.core.config import settings

def test_config_loads():
    assert settings.APP_NAME == "MetaHRM Face Recognition Service"
    assert settings.FACENET_MODEL in ['vggface2', 'casia-webface']
    assert len(settings.mtcnn_thresholds_list) == 3
    assert settings.TARGET_FACE_SIZE == 160
    print("✅ Config loaded successfully!")

if __name__ == "__main__":
    test_config_loads()