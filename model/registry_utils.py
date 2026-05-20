import json
from pathlib import Path
from typing import Optional

from utils.config import CROPPED_DATASET_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "model" / "disease_registry.json"


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        # minimal fallback
        return {
            "version": 1,
            "canonical_labels": [],
            "ai_aliases": {},
        }
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def save_registry(registry: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def normalize_key(s: str) -> str:
    return (s or "").strip().lower()


def resolve_to_canonical(ai_disease: str) -> Optional[str]:
    """Map AI disease string to canonical label.

    If not found in ai_aliases, we return sanitized title-cased label.
    We intentionally do NOT create folders here.
    """
    if not ai_disease:
        return None

    registry = load_registry()

    raw_key = normalize_key(ai_disease)
    alias = registry.get("ai_aliases", {}).get(raw_key)
    if alias:
        return alias

    # If AI already returned canonical exact match (case-insensitive)
    canonical = registry.get("canonical_labels", [])
    for c in canonical:
        if raw_key == normalize_key(c):
            return c

    # Fallback: Title-case the string to form a candidate label.
    # Caller may create folder + add to registry.
    return ai_disease.strip()


def ensure_label_exists(canonical_label: str) -> str:
    """Ensure canonical_label exists in registry and on disk under CroppedData."""
    registry = load_registry()

    canonical_labels = registry.setdefault("canonical_labels", [])
    canonical_label = (canonical_label or "").strip()
    if not canonical_label:
        raise ValueError("canonical_label is empty")

    if canonical_label not in canonical_labels:
        canonical_labels.append(canonical_label)

    # IMPORTANT: preserve append-only stable ordering.
    # Do NOT sort here; incremental training assumes class index stability.
    registry["canonical_labels"] = canonical_labels


    # create folder on disk
    (CROPPED_DATASET_PATH / canonical_label).mkdir(parents=True, exist_ok=True)

    save_registry(registry)
    return canonical_label


def ensure_alias_exists(ai_raw: str, canonical_label: str) -> None:
    registry = load_registry()
    ai_raw = ai_raw or ""
    if not ai_raw.strip():
        return

    aliases = registry.setdefault("ai_aliases", {})
    aliases[normalize_key(ai_raw)] = canonical_label
    save_registry(registry)

