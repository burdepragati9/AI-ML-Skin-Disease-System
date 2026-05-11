import argparse
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "CleanData"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "CroppedData"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMG_SIZE = 224


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def crop_dataset(input_dir: Path, output_dir: Path) -> tuple[int, int]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    skipped = 0

    for class_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        target_dir = output_dir / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)

        for source_path in sorted(class_dir.iterdir()):
            if not source_path.is_file() or source_path.suffix.lower() not in IMAGE_EXTENSIONS:
                skipped += 1
                continue

            try:
                with Image.open(source_path) as pil_image:
                    image = pil_image.convert("RGB")
                    image = center_crop_square(image)
                    image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
                    image.save(target_dir / f"{source_path.stem}.jpg", quality=95)
                    saved += 1

            except Exception as exc:
                skipped += 1
                print(f"Skipped {source_path}: {exc}")

    return saved, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Center-crop skin image dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    saved, skipped = crop_dataset(args.input.resolve(), args.output.resolve())
    print(f"Cropped dataset ready: saved {saved}, skipped {skipped}")


if __name__ == "__main__":
    main()
