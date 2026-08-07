import mediapipe as mp

print(f'MediaPipe version: {mp.__version__}')

# Try to import the Tasks API
try:
    from mediapipe.tasks import python
    print('Successfully imported mediapipe.tasks.python')
    
    from mediapipe.tasks.python import vision
    print('Successfully imported mediapipe.tasks.python.vision')
    
    print(f'Vision module: {dir(vision)}')
    
    # Check for FaceDetector
    if hasattr(vision, 'FaceDetector'):
        print('FaceDetector is available in vision module')
    else:
        print('FaceDetector NOT found in vision module')
        
    # Check for FaceDetectorOptions
    if hasattr(vision, 'FaceDetectorOptions'):
        print('FaceDetectorOptions is available in vision module')
    else:
        print('FaceDetectorOptions NOT found in vision module')
        
except Exception as e:
    print(f'Error importing Tasks API: {e}')
    import traceback
    traceback.print_exc()
