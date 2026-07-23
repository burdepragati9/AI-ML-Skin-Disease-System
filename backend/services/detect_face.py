"""
Face Detection Helper using RetinaFace (via InsightFace)
"""
import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional
import cv2

# Singleton detector instance
_face_detector: Optional['FaceAnalysis'] = None
_detector_initialized = False
_detector_init_error = None


def is_face_complete(face_bbox: Tuple[int, int, int, int], image_width: int, image_height: int) -> bool:
    """
    Validate if the detected face is sufficiently visible for consent purposes.
    
    A face is considered valid if:
    1. The face is not excessively cropped (allows some boundary contact)
    2. The face is sufficiently large relative to the image
    3. The face is not the entire image (likely false positive)
    
    This is more permissive than strict "complete face" detection to allow
    cropped/partial faces while still filtering out tiny detections.
    
    Args:
        face_bbox: Tuple of (x1, y1, x2, y2) bounding box coordinates
        image_width: Width of the image
        image_height: Height of the image
        
    Returns:
        bool: True if face is sufficiently visible, False otherwise
    """
    x1, y1, x2, y2 = face_bbox
    face_width = x2 - x1
    face_height = y2 - y1
    
    print(f"[Face Validation] Face bbox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    print(f"[Face Validation] Face size: {face_width} x {face_height}")
    print(f"[Face Validation] Image size: {image_width} x {image_height}")
    
    # Check if face is excessively cropped (allow some boundary contact)
    # Reduced margin from 5px to 2px to be more permissive
    margin = 2  # 2 pixel margin for tolerance
    touches_left = x1 <= margin
    touches_right = x2 >= (image_width - margin)
    touches_top = y1 <= margin
    touches_bottom = y2 >= (image_height - margin)
    
    print(f"[Face Validation] Touches left: {touches_left}")
    print(f"[Face Validation] Touches right: {touches_right}")
    print(f"[Face Validation] Touches top: {touches_top}")
    print(f"[Face Validation] Touches bottom: {touches_bottom}")
    
    # Count how many boundaries the face touches
    boundary_touches = sum([touches_left, touches_right, touches_top, touches_bottom])
    print(f"[Face Validation] Number of boundary touches: {boundary_touches}")
    
    # If face touches 3 or 4 boundaries, it's likely too cropped
    if boundary_touches >= 3:
        print(f"[Face Validation] Face touches {boundary_touches} boundaries - too cropped")
        return False
    
    # Check if face is sufficiently large relative to image
    # Reduced from 10% to 5% to be more permissive
    min_face_width = image_width * 0.05
    min_face_height = image_height * 0.05
    
    print(f"[Face Validation] Minimum required face size: {min_face_width:.1f} x {min_face_height:.1f}")
    
    if face_width < min_face_width or face_height < min_face_height:
        print(f"[Face Validation] Face too small relative to image")
        return False
    
    # Check if face is too large (likely false positive or whole image is face)
    # Increased from 95% to 98% to be more permissive
    max_face_width = image_width * 0.98
    max_face_height = image_height * 0.98
    
    if face_width > max_face_width or face_height > max_face_height:
        print(f"[Face Validation] Face too large relative to image - possible false positive")
        return False
    
    print(f"[Face Validation] Face is sufficiently visible")
    return True


def _get_detector():
    """
    Get or initialize the singleton RetinaFace detector.
    
    Returns:
        FaceAnalysis: The initialized detector instance
        
    Raises:
        RuntimeError: If detector initialization fails
    """
    global _face_detector, _detector_initialized, _detector_init_error
    
    if _detector_initialized:
        if _detector_init_error:
            raise RuntimeError(f"RetinaFace detector initialization failed previously: {_detector_init_error}")
        if _face_detector is not None:
            return _face_detector
    
    print("[RetinaFace Detection] Initializing detector (singleton)")
    
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        _face_detector = app
        _detector_initialized = True
        _detector_init_error = None
        
        print("[RetinaFace Detection] Model loaded successfully")
        return app
    except Exception as e:
        _detector_initialized = True
        _detector_init_error = str(e)
        print(f"[RetinaFace Detection] Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"RetinaFace detector initialization failed: {str(e)}") from e


def detect_face(image: Union[Image.Image, np.ndarray]) -> bool:
    """
    Detect if a complete/visible human face is present in the image using InsightFace RetinaFace.
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        bool: True if a complete/visible face is detected, False otherwise
        
    Raises:
        RuntimeError: If RetinaFace detector fails to initialize or run
    """
    print("[RetinaFace] ========== FACE DETECTION START ==========")
    print("[RetinaFace] Image received")
    print(f"[RetinaFace] Image type: {type(image)}")
    
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        print(f"[RetinaFace] Converting PIL Image to numpy array")
        print(f"[RetinaFace] PIL Image mode: {image.mode}")
        print(f"[RetinaFace] PIL Image size: {image.size}")
        image_array = np.array(image)
    else:
        print(f"[RetinaFace] Image is already numpy array")
        image_array = image
    
    print(f"[RetinaFace] Initial array shape: {image_array.shape}")
    print(f"[RetinaFace] Initial array dtype: {image_array.dtype}")
    
    # Ensure RGB format first
    if len(image_array.shape) == 2:
        # Grayscale to RGB
        print("[RetinaFace] Converting grayscale to RGB")
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 4:
        # RGBA to RGB
        print("[RetinaFace] Converting RGBA to RGB")
        image_array = image_array[:, :, :3]
    
    # Convert RGB to BGR (InsightFace/RetinaFace expects BGR like OpenCV)
    print("[RetinaFace] Converting RGB to BGR for InsightFace")
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    height, width = image_array.shape[:2]
    print(f"[RetinaFace] Image width: {width}")
    print(f"[RetinaFace] Image height: {height}")
    print(f"[RetinaFace] Image shape: {image_array.shape}")
    print(f"[RetinaFace] Image dtype: {image_array.dtype}")
    
    try:
        # Get singleton detector
        app = _get_detector()
        
        print("[RetinaFace] Detection started")
        # Run face detection
        faces = app.get(image_array)
        
        print(f"[RetinaFace] Raw detections type: {type(faces)}")
        print(f"[RetinaFace] Raw detections: {faces}")
        
        if faces is None or len(faces) == 0:
            print(f"[RetinaFace] Number of detected faces: 0")
            print(f"[RetinaFace] No faces detected by RetinaFace")
            print(f"[RetinaFace] Final face_detected: false")
            print("[RetinaFace] ========== FACE DETECTION END ==========")
            return False
        
        num_faces = len(faces)
        print(f"[RetinaFace] Number of detected faces: {num_faces}")
        
        # Check each detected face for completeness
        valid_faces_count = 0
        for face_idx, face_data in enumerate(faces):
            print(f"[RetinaFace] Processing face {face_idx}")
            
            # InsightFace returns bbox as [x1, y1, x2, y2]
            bbox = face_data.bbox
            print(f"[RetinaFace] Face {face_idx} bounding box: {bbox}")
            
            # Log confidence score if available
            if hasattr(face_data, 'det_score'):
                print(f"[RetinaFace] Face {face_idx} confidence score: {face_data.det_score}")
            else:
                print(f"[RetinaFace] Face {face_idx} confidence score: N/A (attribute not available)")
            
            # Validate if face is complete
            is_complete = is_face_complete((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])), width, height)
            print(f"[RetinaFace] Face {face_idx} is_complete: {is_complete}")
            
            if is_complete:
                valid_faces_count += 1
                print(f"[RetinaFace] Face {face_idx} is complete and visible")
            else:
                print(f"[RetinaFace] Face {face_idx} is partial/cropped or invalid")
        
        print(f"[RetinaFace] Valid faces after filtering: {valid_faces_count}")
        
        # No complete face found
        if valid_faces_count > 0:
            print(f"[RetinaFace] Final face_detected: true")
            print("[RetinaFace] ========== FACE DETECTION END ==========")
            return True
        else:
            print(f"[RetinaFace] Final face_detected: false")
            print("[RetinaFace] ========== FACE DETECTION END ==========")
            return False
        
    except Exception as e:
        error_msg = f"[RetinaFace] ERROR: Face detection failed: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        print("[RetinaFace] ========== FACE DETECTION END (ERROR) ==========")
        # Raise the error instead of silently returning False
        raise RuntimeError(f"RetinaFace detection failed: {str(e)}")
