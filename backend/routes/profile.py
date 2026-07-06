from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from backend.security.dependencies import get_current_user, require_role
from database.db import fetch_one, execute, utc_now


router = APIRouter()

# All endpoints in this file are doctor-only.
doctor_guard = require_role("doctor")


class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    specialization: Optional[str] = None
    clinic_name: Optional[str] = None
    phone: Optional[str] = None
    experience: Optional[int] = None
    location: Optional[str] = None


@router.get("/me", dependencies=[Depends(doctor_guard)])
def get_profile(user=Depends(get_current_user)):
    """Get the current doctor's profile information."""
    doctor_id = int(user["id"])
    
    doctor = fetch_one(
        "SELECT id, full_name, doctor_id, specialization, clinic_name, email, phone, "
        "profile_photo, experience, location, created_at, updated_at FROM doctors WHERE id = ?",
        (doctor_id,)
    )
    
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor profile not found")
    
    return {
        "id": doctor["id"],
        "full_name": doctor["full_name"],
        "doctor_id": doctor["doctor_id"],
        "specialization": doctor["specialization"],
        "clinic_name": doctor["clinic_name"],
        "email": doctor["email"],
        "phone": doctor["phone"],
        "profile_photo": doctor["profile_photo"],
        "experience": doctor["experience"],
        "location": doctor["location"],
        "created_at": doctor["created_at"],
        "updated_at": doctor["updated_at"],
    }


@router.put("/update", dependencies=[Depends(doctor_guard)])
def update_profile(profile_data: ProfileUpdate, user=Depends(get_current_user)):
    """Update the current doctor's profile information."""
    doctor_id = int(user["id"])
    
    # Check if doctor exists
    existing = fetch_one("SELECT id FROM doctors WHERE id = ?", (doctor_id,))
    if not existing:
        raise HTTPException(status_code=404, detail="Doctor profile not found")
    
    # Build dynamic update query
    update_fields = []
    params = []
    
    if profile_data.full_name is not None:
        update_fields.append("full_name = ?")
        params.append(profile_data.full_name)
    
    if profile_data.specialization is not None:
        update_fields.append("specialization = ?")
        params.append(profile_data.specialization)
    
    if profile_data.clinic_name is not None:
        update_fields.append("clinic_name = ?")
        params.append(profile_data.clinic_name)
    
    if profile_data.phone is not None:
        update_fields.append("phone = ?")
        params.append(profile_data.phone)
    
    if profile_data.experience is not None:
        update_fields.append("experience = ?")
        params.append(profile_data.experience)
    
    if profile_data.location is not None:
        update_fields.append("location = ?")
        params.append(profile_data.location)
    
    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    # Add updated_at timestamp
    update_fields.append("updated_at = ?")
    params.append(utc_now())
    
    # Add doctor_id for WHERE clause
    params.append(doctor_id)
    
    query = f"UPDATE doctors SET {', '.join(update_fields)} WHERE id = ?"
    execute(query, tuple(params))
    
    # Fetch and return updated profile
    return get_profile(user)
