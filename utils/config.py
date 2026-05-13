import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def env_path(name: str, default: str) -> Path:
    """Resolve a path setting relative to the project root when needed."""
    value = os.getenv(name, default)
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


MODEL_PATH = env_path("SKIN_MODEL_PATH", "model/skin_model.keras")
CLASS_NAMES_PATH = env_path("CLASS_NAMES_PATH", "model/class_names.json")
DATASET_PATH = env_path("SELF_LEARNING_DATASET_PATH", "dataset")
DB_PATH = env_path("APP_DB_PATH", "database/app.db")
UPLOAD_HISTORY_PATH = env_path("UPLOAD_HISTORY_PATH", "history/uploads")
PROFILE_PHOTO_PATH = env_path("PROFILE_PHOTO_PATH", "history/profile_photos")
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "70.0"))
FREE_SEARCH_LIMIT = int(os.getenv("FREE_SEARCH_LIMIT", "3"))
IMAGE_DUPLICATE_HASH_DISTANCE = int(os.getenv("IMAGE_DUPLICATE_HASH_DISTANCE", "4"))
INCREMENTAL_TRAINING_EPOCHS = int(os.getenv("INCREMENTAL_TRAINING_EPOCHS", "2"))
TRAINING_MAX_ATTEMPTS = int(os.getenv("TRAINING_MAX_ATTEMPTS", "3"))
TRAINING_STALE_MINUTES = int(os.getenv("TRAINING_STALE_MINUTES", "30"))
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "models/gemini-2.5-flash")
GEMINI_API_KEY = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or os.getenv("API_KEY")
)
