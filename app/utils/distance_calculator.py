"""
Distance Calculator
Tính khoảng cách giữa 2 face embeddings
"""

import numpy as np
from scipy.spatial.distance import cosine, euclidean
from typing import Dict
from app.core.config import settings
from app.utils.logger import app_logger


class DistanceCalculator:
    """
    Tính khoảng cách và similarity giữa face embeddings
    """
    
    @staticmethod
    def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Tính khoảng cách Euclidean
        
        Formula: sqrt(sum((a - b)^2))
        
        Threshold: < 1.0 → Same person
        
        Args:
            emb1: Embedding vector 1 (128 dim)
            emb2: Embedding vector 2 (128 dim)
            
        Returns:
            Distance value (0 = identical, higher = more different)
        """
        return float(euclidean(emb1, emb2))
    
    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Tính độ tương đồng Cosine
        
        Formula: 1 - cosine_distance
        
        Threshold: > 0.6 → Same person
        
        Returns:
            Similarity value (0 = different, 1 = identical)
        """
        return float(1 - cosine(emb1, emb2))
    
    @staticmethod
    def is_same_person(
        emb1: np.ndarray,
        emb2: np.ndarray,
        euclidean_threshold: float = None,
        cosine_threshold: float = None
    ) -> Dict:
        """
        Kết hợp cả 2 metrics để quyết định có phải cùng người không
        
        Args:
            emb1: Embedding 1
            emb2: Embedding 2
            euclidean_threshold: Custom threshold (default từ settings)
            cosine_threshold: Custom threshold (default từ settings)
            
        Returns:
            {
                'is_match': bool,
                'euclidean_distance': float,
                'cosine_similarity': float,
                'confidence': float (0-100),
                'method': str
            }
        """
        # Use settings if not provided
        if euclidean_threshold is None:
            euclidean_threshold = settings.EUCLIDEAN_THRESHOLD
        if cosine_threshold is None:
            cosine_threshold = settings.COSINE_THRESHOLD
        
        # Calculate distances
        euc_dist = DistanceCalculator.euclidean_distance(emb1, emb2)
        cos_sim = DistanceCalculator.cosine_similarity(emb1, emb2)
        
        # Decision based on both metrics
        euc_match = euc_dist < euclidean_threshold
        cos_match = cos_sim > cosine_threshold
        
        # Final decision (cả 2 phải pass)
        is_match = euc_match and cos_match
        
        # Calculate confidence score
        # Normalize Euclidean to 0-1 (1 is best)
        # Max expected distance is 2.0, so divide by 2
        euc_score = max(0, 1 - (euc_dist / 2.0))
        
        # Cosine similarity already 0-1
        cos_score = cos_sim
        
        # Weighted average (60% cosine, 40% euclidean)
        confidence = (cos_score * 0.6 + euc_score * 0.4) * 100
        
        # Determine which method contributed more
        if euc_match and cos_match:
            method = "BOTH"
        elif euc_match:
            method = "EUCLIDEAN_ONLY"
        elif cos_match:
            method = "COSINE_ONLY"
        else:
            method = "NONE"
        
        app_logger.debug(
            f"Comparison: euc={euc_dist:.4f} (threshold={euclidean_threshold}), "
            f"cos={cos_sim:.4f} (threshold={cosine_threshold}), "
            f"match={is_match}, confidence={confidence:.2f}%"
        )
        
        return {
            'is_match': is_match,
            'euclidean_distance': round(euc_dist, 4),
            'cosine_similarity': round(cos_sim, 4),
            'confidence': round(confidence, 2),
            'method': method
        }
    
    @staticmethod
    def find_best_match(
        query_embedding: np.ndarray,
        known_embeddings: list
    ) -> Dict:
        """
        T��m embedding khớp nhất trong danh sách
        
        Args:
            query_embedding: Embedding cần so khớp
            known_embeddings: List các embedding đã lưu
            
        Returns:
            {
                'best_match_index': int,
                'best_distance': float,
                'best_similarity': float,
                'is_match': bool,
                'confidence': float,
                'all_comparisons': List[Dict]
            }
        """
        if not known_embeddings:
            return {
                'best_match_index': -1,
                'is_match': False,
                'confidence': 0.0,
                'message': 'No known embeddings to compare'
            }
        
        comparisons = []
        
        for idx, known_emb in enumerate(known_embeddings):
            # Convert to numpy if needed
            if isinstance(known_emb, list):
                known_emb = np.array(known_emb)
            
            result = DistanceCalculator.is_same_person(query_embedding, known_emb)
            result['index'] = idx
            comparisons.append(result)
        
        # Find best match (highest confidence)
        best = max(comparisons, key=lambda x: x['confidence'])
        
        return {
            'best_match_index': best['index'],
            'best_distance': best['euclidean_distance'],
            'best_similarity': best['cosine_similarity'],
            'is_match': best['is_match'],
            'confidence': best['confidence'],
            'method': best['method'],
            'all_comparisons': comparisons
        }