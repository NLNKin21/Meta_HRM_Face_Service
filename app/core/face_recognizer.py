"""
FaceNet Face Recognizer
Generate 128-dim embedding vectors từ ảnh khuôn mặt
"""

import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
from torchvision import transforms
from typing import Optional
from app.core.config import settings
from app.utils.logger import app_logger


class FaceRecognizer:
    """
    FaceNet Inception-ResNet-V1 model
    
    - Input: RGB face image 160x160
    - Output: 128-dim embedding vector
    - Pretrained on VGGFace2 or CASIA-WebFace
    """
    
    def __init__(self):
        """
        Initialize FaceNet model
        
        Model tự động download weights lần đầu (~100MB)
        """
        try:
            # Detect device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            app_logger.info(f"Using device: {self.device}")
            
            # Load pretrained model
            app_logger.info(f"Loading FaceNet model: {settings.FACENET_MODEL}")
            self.model = InceptionResnetV1(
                pretrained=settings.FACENET_MODEL
            ).eval().to(self.device)
            
            app_logger.info("FaceNet model loaded successfully")
            
            # Preprocessing transform
            self.transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
        except Exception as e:
            app_logger.error(f"Failed to initialize FaceNet: {str(e)}")
            raise
    
    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Generate embedding vector từ ảnh khuôn mặt đã crop
        
        Args:
            face_image: numpy array (H, W, 3) RGB
            
        Returns:
            numpy array shape (128,)
        """
        try:
            # Convert to PIL Image
            if not isinstance(face_image, Image.Image):
                pil_image = Image.fromarray(face_image.astype('uint8'))
            else:
                pil_image = face_image
            
            # Ensure RGB
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Preprocess
            tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            app_logger.debug(f"Input tensor shape: {tensor.shape}")
            
            # Get embedding
            with torch.no_grad():
                embedding = self.model(tensor)
            
            # Convert to numpy
            embedding_np = embedding.cpu().numpy().flatten()
            
            app_logger.debug(f"Embedding shape: {embedding_np.shape}")
            
            return embedding_np
            
        except Exception as e:
            app_logger.error(f"Failed to generate embedding: {str(e)}")
            raise
    
    def get_embedding_from_raw_image(
        self,
        image: np.ndarray,
        face_detector
    ) -> Optional[np.ndarray]:
        """
        Pipeline đầy đủ: Detect face → Crop → Embedding
        
        Args:
            image: Ảnh gốc RGB
            face_detector: Instance của FaceDetector
            
        Returns:
            embedding vector (128,) hoặc None nếu không detect được
        """
        try:
            # Detect and crop face
            result = face_detector.get_largest_face(image)
            
            if result is None:
                app_logger.warning("No face detected in image")
                return None
            
            cropped_face, face_info = result
            
            # Generate embedding
            embedding = self.get_embedding(cropped_face)
            
            app_logger.debug(
                f"Generated embedding from face with confidence {face_info['confidence']:.2f}"
            )
            
            return embedding
            
        except Exception as e:
            app_logger.error(f"Failed to get embedding from raw image: {str(e)}")
            return None
    
    def batch_get_embeddings(self, face_images: list) -> list:
        """
        Generate embeddings cho nhiều ảnh (batch processing)
        
        Args:
            face_images: List of numpy arrays
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for face in face_images:
            try:
                emb = self.get_embedding(face)
                embeddings.append(emb)
            except Exception as e:
                app_logger.error(f"Failed to process face in batch: {str(e)}")
                embeddings.append(None)
        
        return embeddings
    
    def get_model_info(self) -> dict:
        """
        Lấy thông tin model để logging/debugging
        """
        return {
            'model_name': 'InceptionResnetV1',
            'pretrained_on': settings.FACENET_MODEL,
            'embedding_dim': 512,
            'input_size': '160x160',
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.model.parameters())
        }