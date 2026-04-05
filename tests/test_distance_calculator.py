# File: tests/test_distance_calculator.py
import numpy as np
from app.utils.distance_calculator import DistanceCalculator

def test_same_person():
    """Test với 2 embedding giống nhau"""
    emb1 = np.random.rand(128)
    emb2 = emb1.copy()
    
    result = DistanceCalculator.is_same_person(emb1, emb2)
    assert result['is_match'] == True
    assert result['confidence'] > 95.0
    print(f"✅ Same person test: confidence={result['confidence']:.2f}%")

def test_different_person():
    """Test với 2 embedding khác nhau"""
    emb1 = np.random.rand(128)
    emb2 = np.random.rand(128)
    
    result = DistanceCalculator.is_same_person(emb1, emb2)
    print(f"Different person: is_match={result['is_match']}, confidence={result['confidence']:.2f}%")

def test_find_best_match():
    """Test tìm match tốt nhất"""
    query = np.random.rand(128)
    known = [
        np.random.rand(128),  # Different
        query.copy(),  # Same
        np.random.rand(128)   # Different
    ]
    
    result = DistanceCalculator.find_best_match(query, known)
    assert result['best_match_index'] == 1
    assert result['is_match'] == True
    print(f"✅ Best match found at index {result['best_match_index']}")

if __name__ == "__main__":
    test_same_person()
    test_different_person()
    test_find_best_match()