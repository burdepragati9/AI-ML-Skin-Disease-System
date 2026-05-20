import json
from pathlib import Path
from typing import Tuple, Dict, List

from utils.config import CROPPED_DATASET_PATH


def _scan_disease_folders(cropped_root: Path) -> List[str]:
    if not cropped_root.exists():
        return []

    folders: List[str] = []
    for p in cropped_root.iterdir():
        if p.is_dir():
            # Canonical folder name is the folder name on disk.
            folders.append(p.name)

    # Stable ordering: alphabetical (case-insensitive) to keep indices deterministic.
    folders.sort(key=lambda s: s.casefold())
    return folders


def build_class_names_and_indices(
    cropped_root: Path = CROPPED_DATASET_PATH,
    class_names_path: Path | None = None,
    class_indices_path: Path | None = None,
) -> Tuple[List[str], Dict[str, int], Dict[str, str]]:
    """Build deterministic mapping from CroppedData folder names.

    Returns:
      - class_names: list of disease folder names ordered by index
      - class_indices: dict disease -> index
      - index_to_label: dict index(str) -> disease
    """

    class_names = _scan_disease_folders(cropped_root)
    class_indices = {name: i for i, name in enumerate(class_names)}
    index_to_label = {str(i): name for i, name in enumerate(class_names)}

    if class_names_path is not None:
        class_names_path.parent.mkdir(parents=True, exist_ok=True)
        class_names_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    if class_indices_path is not None:
        class_indices_path.parent.mkdir(parents=True, exist_ok=True)
        class_indices_path.write_text(json.dumps(class_indices, indent=2), encoding="utf-8")

    return class_names, class_indices, index_to_label


def load_class_names(path: Path) -> List[str]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected class_names.json to contain a JSON list")
    return data


def load_class_indices(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected class_indices.json to contain a JSON object")
    return {str(k): int(v) for k, v in data.items()}

