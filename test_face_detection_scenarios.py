import sys
sys.path.insert(0, 'backend')

from backend.services.detect_face import detect_face
from PIL import Image
import os

print("=" * 80)
print("TESTING FACE DETECTION SCENARIOS")
print("=" * 80)

# Test cases
test_cases = [
    {
        "name": "A. Clear frontal face (Acne image with face)",
        "path": "MySkinData/Acne/acne-pustular-9.jpeg",
        "expected_detector": "retinaface",
        "expected_result": True
    },
    {
        "name": "B. Side-profile face (if available)",
        "path": "MySkinData/Acne/acne-pustular-8.jpeg",
        "expected_detector": "retinaface or mediapipe_tasks_fallback",
        "expected_result": True
    },
    {
        "name": "C. Partial/cropped face (if available)",
        "path": "MySkinData/Acne/acne-pustular-7.jpeg",
        "expected_detector": "retinaface or mediapipe_tasks_fallback",
        "expected_result": True
    },
    {
        "name": "D. Small face (if available)",
        "path": "MySkinData/Acne/acne-pustular-6.jpeg",
        "expected_detector": "retinaface or mediapipe_tasks_fallback",
        "expected_result": True
    },
    {
        "name": "E. Face near image boundary (if available)",
        "path": "MySkinData/Acne/acne-pustular-5.jpeg",
        "expected_detector": "retinaface or mediapipe_tasks_fallback",
        "expected_result": True
    },
    {
        "name": "F. No-face skin lesion image",
        "path": "MySkinData/Acne/acne-scar-1.jpeg",
        "expected_detector": "none",
        "expected_result": False
    },
    {
        "name": "G. White image (no face)",
        "path": None,  # Will create programmatically
        "expected_detector": "none",
        "expected_result": False
    }
]

results = []

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{'=' * 80}")
    print(f"TEST CASE {i}: {test_case['name']}")
    print(f"{'=' * 80}")
    
    try:
        if test_case['path']:
            if not os.path.exists(test_case['path']):
                print(f"SKIP: Image not found at {test_case['path']}")
                results.append({
                    "name": test_case['name'],
                    "status": "SKIPPED",
                    "reason": "Image not found"
                })
                continue
            
            image = Image.open(test_case['path'])
            filename = os.path.basename(test_case['path'])
        else:
            # Create white image for test G
            image = Image.new('RGB', (640, 480), color=(255, 255, 255))
            filename = "white_test_image.jpg"
        
        print(f"Image: {filename}")
        print(f"Size: {image.size}")
        print(f"Mode: {image.mode}")
        
        result = detect_face(image, filename=filename)
        
        print(f"\nRESULT: face_detected = {result}")
        
        # Determine which detector was used based on logs
        # We'll parse the last detector used from the output
        # For now, we'll just record the boolean result
        
        results.append({
            "name": test_case['name'],
            "status": "PASSED" if result == test_case['expected_result'] else "FAILED",
            "expected": test_case['expected_result'],
            "actual": result,
            "expected_detector": test_case['expected_detector']
        })
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            "name": test_case['name'],
            "status": "ERROR",
            "error": str(e)
        })

# Summary
print(f"\n{'=' * 80}")
print("TEST SUMMARY")
print(f"{'=' * 80}")

passed = sum(1 for r in results if r.get('status') == 'PASSED')
failed = sum(1 for r in results if r.get('status') == 'FAILED')
skipped = sum(1 for r in results if r.get('status') == 'SKIPPED')
errors = sum(1 for r in results if r.get('status') == 'ERROR')

print(f"Total tests: {len(results)}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print(f"Skipped: {skipped}")
print(f"Errors: {errors}")

for r in results:
    print(f"\n{r['name']}: {r['status']}")
    if r['status'] == 'FAILED':
        print(f"  Expected: {r['expected']}, Actual: {r['actual']}")
    elif r['status'] == 'ERROR':
        print(f"  Error: {r.get('error', 'Unknown')}")
