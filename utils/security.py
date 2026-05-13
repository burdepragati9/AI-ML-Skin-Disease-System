import hashlib
import re
import uuid
from io import BytesIO
from pathlib import Path

import imagehash
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename


ALLOWED_IMAGE_TYPES = {"jpg", "jpeg", "png"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024


def sanitize_text(value: str, max_length: int = 255) -> str:
    cleaned = re.sub(r"[<>]", "", (value or "").strip())
    return cleaned[:max_length]


def safe_disease_slug(value: str) -> str:
    cleaned = sanitize_text(value, 80)
    cleaned = re.sub(r"[^A-Za-z0-9 _-]", "", cleaned).strip()
    return cleaned or "Unknown"


def validate_image_upload(uploaded_file) -> bytes:
    """Validate Streamlit upload content and return bytes for durable use."""
    name = secure_filename(getattr(uploaded_file, "name", "upload.jpg"))
    extension = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if extension not in ALLOWED_IMAGE_TYPES:
        raise ValueError("Only JPG, JPEG, and PNG images are allowed.")

    content = uploaded_file.getvalue()
    if not content or len(content) > MAX_IMAGE_BYTES:
        raise ValueError("Image must be non-empty and smaller than 8 MB.")

    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc

    return content


def image_from_bytes(content: bytes) -> Image.Image:
    return Image.open(BytesIO(content)).convert("RGB")


def image_hash(image: Image.Image) -> str:
    return str(imagehash.phash(image))


def image_hash_distance(left_hash: str, right_hash: str) -> int:
    """Return perceptual-hash distance; smaller values mean more similar images."""
    return imagehash.hex_to_hash(left_hash) - imagehash.hex_to_hash(right_hash)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def save_optimized_image(image: Image.Image, target_dir: Path, prefix: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = secure_filename(f"{prefix}_{uuid.uuid4().hex[:10]}.jpg")
    path = target_dir / filename
    image.convert("RGB").save(path, format="JPEG", quality=88, optimize=True)
    return path
