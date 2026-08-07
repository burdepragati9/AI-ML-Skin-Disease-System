import sys
sys.path.insert(0, 'backend')

from backend.services.detect_face import detect_face
from PIL import Image
import numpy as np

# Test with a simple test image
print("Testing MediaPipe Tasks Face Detection initialization...")

# Create a simple test image (RGB)
test_image = Image.new('RGB', (640, 480), color=(255, 255, 255))

try:
    result = detect_face(test_image, filename="test_white_image.jpg")
    print(f"Detection result: {result}")
except Exception as e:
    print(f"Error during detection: {e}")
    import traceback
    traceback.print_exc()
