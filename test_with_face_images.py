import sys
sys.path.insert(0, 'backend')

from backend.services.detect_face import detect_face
from PIL import Image
import os

print("=" * 80)
print("TESTING FACE DETECTION WITH ACTUAL FACE IMAGES")
print("=" * 80)

# Test with doctor profile photos (these should contain faces)
test_images = [
    "history/profile_photos/doctor_2_a242688c10.jpg",
    "history/profile_photos/doctor_3_19fa8ba62d.jpg",
    "history/profile_photos/doctor_3_aceb258bce.jpg"
]

for img_path in test_images:
    if not os.path.exists(img_path):
        print(f"\nSKIP: {img_path} not found")
        continue
    
    print(f"\n{'=' * 80}")
    print(f"Testing: {img_path}")
    print(f"{'=' * 80}")
    
    try:
        image = Image.open(img_path)
        filename = os.path.basename(img_path)
        
        print(f"Image: {filename}")
        print(f"Size: {image.size}")
        print(f"Mode: {image.mode}")
        
        result = detect_face(image, filename=filename)
        
        print(f"\nRESULT: face_detected = {result}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
