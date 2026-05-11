import argparse
from pathlib import Path

import cv2
import imagehash
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "MySkinData"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "CleanData"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_SIZE = 100
BLUR_THRESHOLD = 50


def is_blurry(image) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD


def clean_dataset(input_dir: Path, output_dir: Path) -> tuple[int, int]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = set()
    kept = 0
    skipped = 0

    for class_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        target_dir = output_dir / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)

        for source_path in sorted(class_dir.iterdir()):
            if not source_path.is_file() or source_path.suffix.lower() not in IMAGE_EXTENSIONS:
                skipped += 1
                continue

            try:
                cv_image = cv2.imread(str(source_path))
                if cv_image is None:
                    skipped += 1
                    continue

                height, width = cv_image.shape[:2]
                if height < MIN_SIZE or width < MIN_SIZE or is_blurry(cv_image):
                    skipped += 1
                    continue

                with Image.open(source_path) as pil_image:
                    image = pil_image.convert("RGB")
                    image_hash = imagehash.average_hash(image)

                    if image_hash in hashes:
                        skipped += 1
                        continue

                    hashes.add(image_hash)
                    image.save(target_dir / f"{source_path.stem}.jpg", quality=95)
                    kept += 1

            except Exception as exc:
                skipped += 1
                print(f"Skipped {source_path}: {exc}")

    return kept, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean skin image dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    kept, skipped = clean_dataset(args.input.resolve(), args.output.resolve())
    print(f"Clean dataset ready: kept {kept}, skipped {skipped}")


if __name__ == "__main__":
    main()
