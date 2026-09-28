from fastapi import APIRouter, HTTPException, Depends

from backend.security.jwt import create_access_token
from backend.schemas.auth import (
    LoginRequest,
    LoginResponse,
    SignupRequest,
    ForgotPasswordRequest,
    ForgotPasswordResponse,
    ResetPasswordRequest,
    MeResponse,
)
from backend.services.doctor_auth import (
    authenticate_doctor,
    create_doctor,
    create_reset_token,
    reset_password,
)
from backend.services.admin_auth import admin_authenticate
from backend.security.dependencies import get_current_user
from utils.queries import GET_DOCTOR_PROFILE_BY_ID


router = APIRouter()


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest):
    email = payload.email
    password = payload.password

    # Doctor auth first
    doctor = authenticate_doctor(email, password)
    if doctor:
        access_token = create_access_token(
            user_id=int(doctor["id"]),
            email=doctor["email"],
            role="doctor",
        )
        return {
            "success": True,
            "role": "doctor",
            "access_token": access_token,
            "user": {"id": int(doctor["id"]), "email": doctor["email"]},
        }

    # Fallback to admin auth
    admin = admin_authenticate(email, password)
    if admin:
        access_token = create_access_token(
            user_id=int(admin["id"]),
            email=admin["email"],
            role="admin",
        )
        return {
            "success": True,
            "role": "admin",
            "access_token": access_token,
            "user": {"id": int(admin["id"]), "email": admin["email"]},
        }

    raise HTTPException(status_code=401, detail="Invalid email or password")


@router.post("/signup")
def signup(payload: SignupRequest):
    import sqlite3

    try:
        doctor_pk = create_doctor(payload.model_dump(), payload.password)
    except sqlite3.IntegrityError:
        # UNIQUE constraint failed (e.g., doctors.email)
        raise HTTPException(status_code=400, detail="Email already registered.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"success": True, "doctor_id": doctor_pk}



@router.post("/forgot-password", response_model=ForgotPasswordResponse)
def forgot_password(payload: ForgotPasswordRequest):
    token = create_reset_token(payload.email)
    # For security: do not reveal whether email exists.
    # BUT your requirement asks to show token in response.
    if not token:
        # Still comply with contract shape.
        return ForgotPasswordResponse(success=True, token="")
    return ForgotPasswordResponse(success=True, token=token)


@router.post("/reset-password")
def reset_password_route(payload: ResetPasswordRequest):
    try:
        ok = reset_password(payload.token, payload.new_password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not ok:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    return {"success": True}


@router.get("/me", response_model=MeResponse)
def me(user=Depends(get_current_user)):
    # If user is a doctor, fetch additional doctor details from database
    if user.get("role") == "doctor":
        from database.db import fetch_one
        doctor_row = fetch_one(
            GET_DOCTOR_PROFILE_BY_ID,
            (int(user["id"]),)
        )
        print("Doctor Row:", doctor_row)
        doctor_data = dict(doctor_row) if doctor_row else {}
        print("Doctor Data:", doctor_data)

        if doctor_data:
            # Merge doctor details with user info
            return {
                **user,
                "full_name": doctor_data.get("full_name"),
                "specialization": doctor_data.get("specialization"),
                "clinic_name": doctor_data.get("clinic_name"),
                "phone": doctor_data.get("phone"),
                "experience": doctor_data.get("experience"),
                "location": doctor_data.get("location"),
            }
    return user




