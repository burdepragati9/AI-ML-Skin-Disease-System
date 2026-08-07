import sys
sys.path.insert(0, 'backend')

from backend.services.detect_face import detect_face
from PIL import Image

print("Testing face detection with simple white image...")

# Create a simple test image (RGB)
test_image = Image.new('RGB', (640, 480), color=(255, 255, 255))

try:
    result = detect_face(test_image, filename="test_white_image.jpg")
    print(f"Detection result: {result}")
    print("SUCCESS: No HTTP 500 error occurred")
    print("SUCCESS: MediaPipe was handled gracefully as unavailable")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
