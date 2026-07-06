from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field
from typing import Optional



class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=200)


class LoginUser(BaseModel):
    id: int
    email: EmailStr


class LoginResponse(BaseModel):
    success: bool
    role: str
    access_token: str
    user: LoginUser


class SignupRequest(BaseModel):
    full_name: str
    specialization: str
    clinic_name: str | None = None
    email: EmailStr
    phone: str | None = None
    experience: int = 0
    location: str | None = None
    password: str = Field(min_length=8, max_length=200)


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ForgotPasswordResponse(BaseModel):
    success: bool
    token: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(min_length=8, max_length=200)


class MeResponse(BaseModel):
    id: int
    email: EmailStr
    role: str
    full_name: Optional[str] = None
    specialization: Optional[str] = None
    clinic_name: Optional[str] = None
    phone: Optional[str] = None
    experience: Optional[int] = None
    location: Optional[str] = None



