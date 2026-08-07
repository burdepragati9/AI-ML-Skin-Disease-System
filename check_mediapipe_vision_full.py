import mediapipe as mp

print(f'MediaPipe version: {mp.__version__}')

if hasattr(mp, 'tasks'):
    if hasattr(mp.tasks, 'python'):
        if hasattr(mp.tasks.python, 'vision'):
            vision = mp.tasks.python.vision
            print(f'Vision module contents: {dir(vision)}')
            
            # Check for FaceDetector
            if hasattr(vision, 'FaceDetector'):
                print('FaceDetector is available')
                print(f'FaceDetector contents: {dir(vision.FaceDetector)}')
            
            # Check for FaceDetectorOptions
            if hasattr(vision, 'FaceDetectorOptions'):
                print('FaceDetectorOptions is available')
