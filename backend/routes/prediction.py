from fastapi import APIRouter, UploadFile, File, Depends

from PIL import Image

from backend.services.prediction_service import predict_disease
from backend.security.dependencies import get_current_user, require_role
from history.search_history import record_search


router = APIRouter()

# Predictions are doctor-only (but we do not change auth logic elsewhere).
doctor_guard = require_role("doctor")


@router.get("/")
def health_check():
    return {"status": "running"}


@router.post("/predict", dependencies=[Depends(doctor_guard)])
async def predict(file: UploadFile = File(...), user=Depends(get_current_user)):
    print("[predict endpoint] reached")


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
    result = predict_disease(image)

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
        )

    # Preserve existing prediction response format.
    return result
