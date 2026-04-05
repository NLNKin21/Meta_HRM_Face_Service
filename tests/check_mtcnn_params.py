"""Check MTCNN available parameters"""
from mtcnn import MTCNN
import inspect

# Check MTCNN __init__ signature
sig = inspect.signature(MTCNN.__init__)
print("MTCNN.__init__ parameters:")
for param_name, param in sig.parameters.items():
    if param_name != 'self':
        print(f"  - {param_name}: {param.default if param.default != inspect.Parameter.empty else 'required'}")

# Try to initialize
try:
    detector = MTCNN()
    print("\n✅ MTCNN initialized successfully with default params")
except Exception as e:
    print(f"\n❌ Error: {e}")

try:
    detector = MTCNN(min_face_size=20)
    print("✅ MTCNN initialized with min_face_size=20")
except Exception as e:
    print(f"❌ Error: {e}")