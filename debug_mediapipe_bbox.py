import sys
sys.path.insert(0, 'backend')

from backend.services.detect_face import detect_face
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Test MediaPipe Tasks bounding box structure
print("Testing MediaPipe Tasks bounding box structure...")

# Create a simple test image
test_image = Image.new('RGB', (640, 480), color=(255, 255, 255))
image_array = np.array(test_image)

# Initialize MediaPipe Tasks
model_path = "backend/models/mediapipe/blaze_face_short_range.tflite"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.5
)
detector = vision.FaceDetector.create_from_options(options)

# Create mp.Image
image_rgb_contiguous = np.ascontiguousarray(image_array)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb_contiguous)

# Run detection
mp_results = detector.detect(mp_image)

if mp_results.detections:
    for det_idx, detection in enumerate(mp_results.detections):
        print(f"\nDetection {det_idx}:")
        print(f"  Detection object: {detection}")
        print(f"  Bounding box: {detection.bounding_box}")
        print(f"  Bounding box type: {type(detection.bounding_box)}")
        
        bbox = detection.bounding_box
        print(f"  origin_x: {bbox.origin_x}")
        print(f"  origin_y: {bbox.origin_y}")
        print(f"  width: {bbox.width}")
        print(f"  height: {bbox.height}")
        
        if hasattr(bbox, '__dict__'):
            print(f"  Bounding box attributes: {bbox.__dict__}")
else:
    print("No detections found")
