from __future__ import annotations

from fastapi import APIRouter, Depends, Query


from backend.security.dependencies import get_current_user, require_role
from history.search_history import doctor_search_stats, recent_searches
from backend.services.doctor_auth import usage_for_doctor



router = APIRouter()

# All endpoints in this file are doctor-only.
doctor_guard = require_role("doctor")


@router.get("/doctor/dashboard/stats", dependencies=[Depends(doctor_guard)])
def get_doctor_dashboard_stats(user=Depends(get_current_user)):
    stats = usage_for_doctor(int(user["id"]))
    return {
        "total_searches": stats.get("used", 0),
        "free_searches_left": stats.get("remaining", 0),
        "used_searches": stats.get("used", 0),
    }



@router.get("/doctor/dashboard/diseases", dependencies=[Depends(doctor_guard)])
def get_doctor_dashboard_diseases(user=Depends(get_current_user)):
    d = doctor_search_stats(int(user["id"]))
    return list(d.get("diseases", []))


@router.get("/doctor/dashboard/prediction-sources", dependencies=[Depends(doctor_guard)])
def get_doctor_dashboard_prediction_sources(user=Depends(get_current_user)):
    d = doctor_search_stats(int(user["id"]))
    return list(d.get("sources", []))


@router.get("/doctor/dashboard/history", dependencies=[Depends(doctor_guard)])
def get_doctor_dashboard_history(
    user=Depends(get_current_user),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    disease: str = Query("", description="Optional disease substring filter"),
):
    offset = (page - 1) * limit
    rows = recent_searches(int(user["id"]), limit=limit, offset=offset, disease=disease or "")
    return {
        "page": page,
        "rows": [
            {
                "disease": r.get("disease"),
                "confidence": r.get("confidence"),
                "prediction_source": r.get("prediction_source"),
                "created_at": r.get("created_at"),
                "image_path": r.get("image_path"),
            }
            for r in (rows or [])
        ],
    }


@router.get("/doctor/dashboard/images", dependencies=[Depends(doctor_guard)])
def get_doctor_dashboard_images(user=Depends(get_current_user)):
    d = doctor_search_stats(int(user["id"]))
    return list(d.get("images", []))


@router.get("/dashboard/doctor", dependencies=[Depends(doctor_guard)])
def get_doctor_dashboard_aggregated(user=Depends(get_current_user)):
    """Aggregated doctor dashboard analytics.

    Must stay doctor-scoped and use the strict 4-search limit.
    """
    doctor_id = int(user["id"])

    # Use existing DB stats but compute free searches left from fixed limit (4).
    d = doctor_search_stats(doctor_id)
    used_searches = int(d.get("total", 0) or 0)
    free_searches_left = max(0, 4 - used_searches)

    # Newest prediction history first.
    history_limit = 10
    rows = recent_searches(doctor_id, limit=history_limit, offset=0, disease="")

    prediction_history = [
        {
            "image_name": r["image_name"] if "image_name" in r.keys() else None,
            "predicted_disease": r["disease"] if "disease" in r.keys() else None,
            "confidence_score": r["confidence"] if "confidence" in r.keys() else None,
            "prediction_source": r["prediction_source"] if "prediction_source" in r.keys() else None,
            "timestamp": r["created_at"] if "created_at" in r.keys() else None,
            "image_path": r["image_path"] if "image_path" in r.keys() else None,
        }
        for r in [dict(rr) if not isinstance(rr, dict) else rr for rr in (rows or [])]
    ]



    # Most searched diseases/images and AI vs ML distribution.
    most_searched_diseases = [
        {
            "disease": x["disease"] if "disease" in x.keys() else None,
            "count": x["count"] if "count" in x.keys() else None,
        }
        for x in [dict(xx) if not isinstance(xx, dict) else xx for xx in (d.get("diseases", []) or [])]
    ]



    ai_vs_ml = [
        {
            "prediction_source": x["prediction_source"] if "prediction_source" in x.keys() else None,
            "count": x["count"] if "count" in x.keys() else None,
        }
        for x in [dict(xx) if not isinstance(xx, dict) else xx for xx in (d.get("sources", []) or [])]
    ]



    def to_upload_url(image_path: str | None) -> str | None:
        """Convert stored filesystem path to a public URL for FastAPI StaticFiles.

        Backward compatible: if it's already a URL, return as-is.
        """
        if not image_path:
            return None

        if isinstance(image_path, str) and (image_path.startswith("http://") or image_path.startswith("https://")):
            return image_path

        # Expected stored format is often Windows absolute path, e.g.
        # C:/Updated_Project/history/uploads/Acne/search_xxx.jpg
        # We map it to:
        # /uploads/Acne/search_xxx.jpg
        marker = "history/uploads/"
        idx = image_path.replace('\\', '/').find(marker)
        if idx == -1:
            # Best-effort: if it's already a relative path like uploads/..., keep it.
            rel = image_path.replace('\\', '/').lstrip('/')
            if rel.startswith('uploads/'):
                return f"http://127.0.0.1:8000/{rel}"
            return None

        rel_part = image_path.replace('\\', '/')[idx + len(marker):]
        return f"http://127.0.0.1:8000/uploads/{rel_part}"

    most_searched_images = []
    for x in (d.get("images", []) or []):
        row = dict(x) if not isinstance(x, dict) else x
        image_path = row.get("image_path")
        image_url = to_upload_url(image_path)
        most_searched_images.append(
            {
                # Backward compatibility (do not remove):
                "image_path": image_path if "image_path" in row.keys() else None,
                # New field used by the frontend:
                "image_url": image_url,
                "disease": row.get("disease"),
                "count": row.get("count"),
                "image_hash": row.get("image_hash"),
            }
        )




    return {
        "total_searches": used_searches,
        "used_searches": used_searches,
        "free_searches_left": free_searches_left,
        "prediction_history": prediction_history,
        "most_searched_diseases": most_searched_diseases,
        "ai_vs_ml": ai_vs_ml,
        "most_searched_images": most_searched_images,
    }


