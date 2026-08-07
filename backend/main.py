from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routes.prediction import router as prediction_router
from backend.routes.auth import router as auth_router
from backend.routes.doctor_dashboard import router as doctor_dashboard_router
from backend.routes.profile import router as profile_router
from backend.routes.admin import router as admin_router
from utils.config import UPLOAD_HISTORY_PATH, PROJECT_ROOT

# Ensure the uploads directory exists before StaticFiles is mounted.
UPLOAD_HISTORY_PATH.mkdir(parents=True, exist_ok=True)
print(f"[main] Upload directory (absolute): {UPLOAD_HISTORY_PATH.resolve()}")
print(f"[main] Upload directory exists: {UPLOAD_HISTORY_PATH.exists()}")


def create_app() -> FastAPI():

    app = FastAPI(title="Skin Disease Detection API")

    allowed_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Ensure CORS preflight requests don't fall through other middleware.

    app.include_router(prediction_router, prefix="/predict")

    app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
    app.include_router(doctor_dashboard_router)
    app.include_router(profile_router, prefix="/profile")
    app.include_router(admin_router, prefix="/admin", tags=["Admin"])

    app.mount(
        "/uploads",
        StaticFiles(directory=str(UPLOAD_HISTORY_PATH)),
        name="uploads",
    )

    return app



app = create_app()

