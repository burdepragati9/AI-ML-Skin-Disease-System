from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from backend.security.jwt import decode_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

# Some axios clients may send an already-parsed/decoded token or non-string value.
# Keep get_current_user tolerant to avoid false “missing auth header” cases.




def get_current_user(token: str = Depends(oauth2_scheme)):
    # TEMP debug logging for auth debugging (doctor_guard / JWT resolution)
    # NOTE: prints appear in the Uvicorn/FastAPI server console.
    auth_header_present = token is not None and token != ""
    print("[doctor_guard] Authorization header present:", auth_header_present)
    print("[doctor_guard] TOKEN TYPE:", type(token))
    print("[doctor_guard] TOKEN VALUE (prefix):", repr(token)[:100])

    if auth_header_present:
        print("[doctor_guard] Raw token (first 24 chars):", token[:24])

    try:
        payload = decode_token(token)
    except Exception as e:
        print("[doctor_guard] Token decode failed reason:", repr(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    # Decode fields for debugging
    decoded_email = payload.get("email")
    decoded_role = payload.get("role")
    decoded_sub = payload.get("sub")
    print("[doctor_guard] Decoded JWT payload:", {"sub": decoded_sub, "email": decoded_email, "role": decoded_role})

    user = {
        "id": int(decoded_sub),
        "email": decoded_email,
        "role": decoded_role,
    }
    print("[doctor_guard] Returning user from get_current_user:", user)

    return user





def require_role(required_role: str):
    def _guard(user=Depends(get_current_user)):
        decoded_role = user.get("role")
        print("[doctor_guard] require_role expected:", required_role, "decoded role:", decoded_role)
        if decoded_role != required_role:
            print("[doctor_guard] Authentication failure reason: insufficient role")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient role permissions",
            )
        return user

    return _guard


