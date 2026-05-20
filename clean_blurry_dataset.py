import argparse
import json
import shutil
from pathlib import Path

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def is_blurry(image_path: Path, threshold: float) -> bool:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return True
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var < threshold


def iter_images(dataset_root: Path):
    for path in dataset_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def load_preflight_paths(report_path: Path) -> list[Path]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    paths: list[Path] = []
    for issue in payload.get("issues", []):
        path = issue.get("path")
        if path:
            paths.append(Path(path))
    return paths


def quarantine_image(image_path: Path, dataset_root: Path, bad_root: Path, *, copy_only: bool) -> Path:
    relative_path = image_path.relative_to(dataset_root)
    target_path = bad_root / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        stem = target_path.stem
        suffix = target_path.suffix
        counter = 1
        while True:
            candidate = target_path.with_name(f"{stem}_{counter}{suffix}")
            if not candidate.exists():
                target_path = candidate
                break
            counter += 1

    if copy_only:
        shutil.copy2(image_path, target_path)
    else:
        shutil.move(str(image_path), str(target_path))
    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Quarantine blurry/low-detail dataset images.")
    parser.add_argument("--dataset", type=Path, default=Path("CroppedData"))
    parser.add_argument("--bad-root", type=Path, default=Path("CroppedData_bad"))
    parser.add_argument("--threshold", type=float, default=100.0)
    parser.add_argument("--from-report", type=Path, default=Path("analytics/dataset_preflight_report.json"))
    parser.add_argument("--scan", action="store_true", help="Scan images with OpenCV instead of using preflight report.")
    parser.add_argument("--copy-only", action="store_true", help="Copy instead of moving images out of the dataset.")
    args = parser.parse_args()

    dataset_root = args.dataset.resolve()
    bad_root = args.bad_root.resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

    # Guardrail: only quarantine images that are inside the selected dataset root.
    candidates: list[Path] = []
    if args.scan:
        candidates = [p for p in iter_images(dataset_root) if is_blurry(p, args.threshold)]
    else:
        if not args.from_report.exists():
            raise FileNotFoundError(f"Preflight report not found: {args.from_report}")
        for relative_or_abs in load_preflight_paths(args.from_report):
            image_path = relative_or_abs
            if not image_path.is_absolute():
                image_path = Path.cwd() / image_path
            image_path = image_path.resolve()
            try:
                image_path.relative_to(dataset_root)
            except ValueError:
                continue
            if image_path.exists() and image_path.suffix.lower() in IMAGE_EXTS:
                candidates.append(image_path)

    moved_by_class: dict[str, int] = {}
    for image_path in candidates:
        relative = image_path.relative_to(dataset_root)
        class_name = relative.parts[0] if relative.parts else "unknown"
        quarantine_image(image_path, dataset_root, bad_root, copy_only=args.copy_only)
        moved_by_class[class_name] = moved_by_class.get(class_name, 0) + 1

    action = "Copied" if args.copy_only else "Moved"
    print(f"{action} {sum(moved_by_class.values())} images to {bad_root}")
    for class_name, count in sorted(moved_by_class.items()):
        print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()
