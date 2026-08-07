from fastapi import APIRouter, UploadFile, File, Depends, Form

from PIL import Image

from backend.services.prediction_service import predict_disease
from backend.services.detect_face import detect_face_details
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
    filename = file.filename or "<unknown>"

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

    if used_searches >= MAX_FREE_SEARCHES:
        from fastapi import HTTPException, status

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You have reached your free prediction limit.",
        )


    image = Image.open(file.file)
    result = predict_disease(image)

    if result.get("status") == "success":
        print(f"[predict] filename={filename} consent={consent_for_training} "
              f"disease={result.get('predicted_disease')} "
              f"confidence={result.get('confidence')} "
              f"source={result.get('prediction_source')}")

    if result.get("status") == "success":
        # Persist prediction history (timestamp handled in record_search via utc_now()).
        # image_path/image_hash are generated inside record_search via utils.security.
        # The image is always saved to disk for display in Reports/History.
        search_id = record_search(
            doctor_pk=doctor_id,
            image=image,
            disease=result.get("predicted_disease"),
            confidence=float(result.get("confidence") or 0.0),
            prediction_source=result.get("prediction_source") or "unknown",
            image_name=file.filename,
            consent_for_training=consent_for_training,
        )
        print(f"[predict] record_search id={search_id} original={file.filename}")

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
    filename = file.filename or "<unknown>"

    try:
        # Read file bytes to get accurate size
        file_bytes = await file.read()
        from io import BytesIO

        image = Image.open(BytesIO(file_bytes))

        # Also check EXIF orientation and apply if needed
        try:
            from PIL import ImageOps
            image = ImageOps.exif_transpose(image)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as exif_err:
            print(f"[detect-face] EXIF transpose skipped: {exif_err}")

        face_result = detect_face_details(image, filename=filename)
        face_detected = bool(face_result["face_detected"])
        detector_used = face_result["detector_used"]
        consent_required = bool(face_result["consent_required"])

        print(f"[detect-face] filename={filename} face_detected={face_detected} "
              f"detector_used={detector_used} consent_required={consent_required}")

        return {
            "success": True,
            "face_detected": face_detected,
            "detector_used": detector_used,
            "consent_required": consent_required,
        }
    except Exception as e:
        print(f"[detect-face] error: {e}")
        import traceback
        traceback.print_exc()
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face detection failed: {str(e)}"
        )