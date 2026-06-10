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
MOBILENET_MODEL_PATH = env_path("MOBILENET_MODEL_PATH", "model/mobilenet_model.keras")
EFFICIENTNET_MODEL_PATH = env_path("EFFICIENTNET_MODEL_PATH", "model/efficientnet_model.keras")
DENSENET_MODEL_PATH = env_path("DENSENET_MODEL_PATH", "model/densenet_model.keras")

# Multiple model support
# Dictionary of model configurations for future extensibility
# Format: {"model_name": {"path": "path/to/model.keras", "class_names": "path/to/class_names.json", "type": "model_type"}}
AVAILABLE_MODELS = {
    "mobilenetv2": {
        "path": MOBILENET_MODEL_PATH if MOBILENET_MODEL_PATH.exists() else MODEL_PATH,
        "class_names": CLASS_NAMES_PATH,
        "type": "mobilenetv2",
        "priority": 10,
    },
    "efficientnetb0": {
        "path": EFFICIENTNET_MODEL_PATH,
        "class_names": CLASS_NAMES_PATH,
        "type": "efficientnetb0",
        "priority": 20,
    },
    "densenet121": {
        "path": DENSENET_MODEL_PATH,
        "class_names": CLASS_NAMES_PATH,
        "type": "densenet121",
        "priority": 30,
    },
}
ACTIVE_MODEL = os.getenv("ACTIVE_MODEL", "mobilenetv2")

# Multi-model prediction configuration
ENABLE_MULTI_MODEL_PREDICTION = os.getenv("ENABLE_MULTI_MODEL_PREDICTION", "true").lower() == "true"
# Self-learning dataset for incremental AI-added images.
# Per requirements, AI-added images must be stored inside CroppedData.
CROPPED_DATASET_PATH = env_path("CROPPED_DATASET_PATH", "CroppedData")

# Backward-compat: keep DATASET_PATH pointing to CroppedData for the existing pipeline.
DATASET_PATH = CROPPED_DATASET_PATH

DB_PATH = env_path("APP_DB_PATH", "database/app.db")
UPLOAD_HISTORY_PATH = env_path("UPLOAD_HISTORY_PATH", "history/uploads")
PROFILE_PHOTO_PATH = env_path("PROFILE_PHOTO_PATH", "history/profile_photos")
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "70.0"))
FREE_SEARCH_LIMIT = int(os.getenv("FREE_SEARCH_LIMIT", "3"))
IMAGE_DUPLICATE_HASH_DISTANCE = int(os.getenv("IMAGE_DUPLICATE_HASH_DISTANCE", "4"))
INCREMENTAL_TRAINING_EPOCHS = int(os.getenv("INCREMENTAL_TRAINING_EPOCHS", "2"))
TRAINING_MAX_ATTEMPTS = int(os.getenv("TRAINING_MAX_ATTEMPTS", "3"))
TRAINING_STALE_MINUTES = int(os.getenv("TRAINING_STALE_MINUTES", "30"))

# Replay incremental learning (to avoid catastrophic forgetting)
# When a new AI-corrected image is added, training will use:
#   - the new image
#   - plus balanced replay samples from existing disease folders.
REPLAY_SAMPLES_PER_DISEASE = int(os.getenv("REPLAY_SAMPLES_PER_DISEASE", "2"))
REPLAY_MAX_TOTAL_SAMPLES = int(os.getenv("REPLAY_MAX_TOTAL_SAMPLES", "20"))
REPLAY_SHUFFLE_SEED = int(os.getenv("REPLAY_SHUFFLE_SEED", "123"))

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "models/gemini-2.5-flash")

GEMINI_API_KEY = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or os.getenv("API_KEY")
)

# AI Verification Configuration
# Enable AI-based prediction validation after ML prediction
ENABLE_AI_VERIFICATION = os.getenv("ENABLE_AI_VERIFICATION", "true").lower() == "true"
# Confidence threshold for triggering AI verification
AI_VERIFICATION_THRESHOLD = float(os.getenv("AI_VERIFICATION_THRESHOLD", "70.0"))
