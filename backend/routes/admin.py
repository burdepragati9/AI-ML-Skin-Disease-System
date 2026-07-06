from fastapi import APIRouter, Depends
from analytics.admin_analytics import admin_summary, ai_recognized_images, training_status
from database.db import fetch_all, fetch_one
from backend.security.dependencies import get_current_user, require_role

router = APIRouter()


@router.get("/analytics", dependencies=[Depends(require_role("admin"))])
def get_admin_analytics(user=Depends(get_current_user)):
    """Get admin analytics data including top metrics, disease counts, source counts, and AI images."""
    summary = admin_summary()
    ai_images = ai_recognized_images(limit=10)
    
    return {
        "ai_images": summary["total_ai"],
        "retrained_images": summary["retrained"],
        "most_added_disease": summary["most_common"]["predicted_disease"] if summary["most_common"] else "None",
        "accuracy_improvement": f"{summary['accuracy_improvement']:.2%}",
        "disease_counts": summary["disease_counts"],
        "source_counts": summary["source_counts"],
        "ai_recognized_images": ai_images,
    }


@router.get("/training-status", dependencies=[Depends(require_role("admin"))])
def get_training_status(user=Depends(get_current_user)):
    """Get training queue status."""
    return training_status()


@router.get("/doctors", dependencies=[Depends(require_role("admin"))])
def get_all_doctors(user=Depends(get_current_user)):
    """Get all registered doctors with their total images analyzed."""
    doctors = fetch_all(
        """
        SELECT 
            id,
            full_name as doctor_name,
            email,
            specialization,
            created_at
        FROM doctors
        ORDER BY created_at DESC
        """
    )
    
    # Calculate total images analyzed for each doctor
    doctors_with_stats = []
    for doctor in doctors:
        doctor_id = doctor["id"]
        total_images = fetch_one(
            """
            SELECT COUNT(*) as count
            FROM searches
            WHERE doctor_id = ?
            """,
            (doctor_id,)
        )["count"]
        
        doctors_with_stats.append({
            "id": doctor["id"],
            "doctor_name": doctor["doctor_name"],
            "email": doctor["email"],
            "specialization": doctor["specialization"],
            "total_images_analyzed": total_images,
            "created_at": doctor["created_at"]
        })
    
    return doctors_with_stats


@router.get("/training-logs", dependencies=[Depends(require_role("admin"))])
def get_training_logs(user=Depends(get_current_user)):
    """Get recent training logs."""
    logs = fetch_all(
        """
        SELECT 
            id,
            event_type,
            status,
            disease,
            accuracy_before,
            accuracy_after,
            message,
            image_path,
            created_at
        FROM training_logs
        ORDER BY created_at DESC
        LIMIT 20
        """
    )
    
    return [
        {
            "id": log["id"],
            "event_type": log["event_type"],
            "status": log["status"],
            "disease": log["disease"],
            "accuracy_before": log["accuracy_before"],
            "accuracy_after": log["accuracy_after"],
            "message": log["message"],
            "image_path": log["image_path"],
            "created_at": log["created_at"]
        }
        for log in logs
    ]
