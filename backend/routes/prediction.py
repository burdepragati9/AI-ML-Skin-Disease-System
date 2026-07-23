from fastapi import APIRouter, UploadFile, File, Depends, Form

from PIL import Image

from backend.services.prediction_service import predict_disease
from backend.services.detect_face import detect_face
from backend.security.dependencies import get_current_user, require_role
from history.search_history import record_search


router = APIRouter()

# Predictions are doctor-only (but we do not change auth logic elsewhere).
doctor_guard = require_role("doctor")


@router.get("/")
def health_check():
    return {"status": "running"}


@router.post("/predict", dependencies=[Depends(doctor_guard)])
async def predict(
    file: UploadFile = File(...),
    consent_for_training: bool = Form(False),
    user=Depends(get_current_user)
):
    print("[predict endpoint] reached")
    print(f"[Prediction] Consent received: {consent_for_training}")

    """Run prediction and persist doctor-specific history with a strict 4-search limit."""
    from database.db import fetch_one

    MAX_FREE_SEARCHES = 4
    doctor_id = int(user["id"])

    # Enforce limit using existing searches table.
    used_searches_row = fetch_one(
        "SELECT COUNT(*) AS c FROM searches WHERE doctor_id = ?",
        (doctor_id,),
    )
    used_searches = int(used_searches_row["c"] or 0) if used_searches_row else 0

    print("[predict limit]")
    print("doctor_id =", doctor_id)
    print("used_searches =", used_searches)
    print("MAX_FREE_SEARCHES =", MAX_FREE_SEARCHES)

    if used_searches >= MAX_FREE_SEARCHES:
        from fastapi import HTTPException, status

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You have reached your free prediction limit.",
        )


    image = Image.open(file.file)
    print("[Prediction] Prediction started")
    result = predict_disease(image)
    print("[Prediction] Prediction completed")

    if result.get("status") == "success":
        # Persist prediction history (timestamp handled in record_search via utc_now()).
        # image_path/image_hash are generated inside record_search via utils.security.
        record_search(
            doctor_pk=doctor_id,
            image=image,
            disease=result.get("predicted_disease"),
            confidence=float(result.get("confidence") or 0.0),
            prediction_source=result.get("prediction_source") or "unknown",
            image_name=file.filename,
            consent_for_training=consent_for_training,
        )

    # Preserve existing prediction response format.
    return result


@router.post("/detect-face", dependencies=[Depends(doctor_guard)])
async def detect_face_endpoint(
    file: UploadFile = File(...),
    user=Depends(get_current_user)
):
    """
    Detect if a complete/visible human face is present in the uploaded image using RetinaFace.
    
    This endpoint is called before prediction to determine if the Face Privacy Consent
    popup should be shown to the user. Only complete, clearly visible faces trigger consent.
    """
    print("[detect-face endpoint] reached")
    print(f"[detect-face endpoint] Filename: {file.filename}")
    print(f"[detect-face endpoint] Content type: {file.content_type}")
    print(f"[detect-face endpoint] File size: {file.size if hasattr(file, 'size') else 'unknown'} bytes")
    
    try:
        image = Image.open(file.file)
        print(f"[detect-face endpoint] Image opened successfully, mode: {image.mode}, size: {image.size}")
        
        face_detected = detect_face(image)
        
        print(f"[detect-face endpoint] Final result: face_detected = {face_detected}")
        
        return {
            "success": True,
            "face_detected": face_detected
        }
    except Exception as e:
        print(f"[detect-face endpoint] Error: {e}")
        import traceback
        traceback.print_exc()
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face detection failed: {str(e)}"
        )
