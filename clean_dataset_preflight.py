import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class ImageIssue:
    path: str
    issue: str
    details: dict


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_basic_stats(path: Path) -> dict:
    # Loads with PIL; raises if image is corrupt
    im = Image.open(path)
    im = im.convert("RGB")
    w, h = im.size

    # Lightweight blur proxy (no OpenCV): use downsample + variance of grayscale
    # (Not perfect Laplacian-variance, but works as a proxy and is dependency-free.)
    gray = im.resize((max(16, w // 32), max(16, h // 32))).convert("L")
    pixels = list(gray.getdata())
    mean = sum(pixels) / len(pixels)
    var = sum((p - mean) ** 2 for p in pixels) / max(1, len(pixels) - 1)

    return {
        "width": w,
        "height": h,
        "min_side": int(min(w, h)),
        "gray_var_proxy": float(var),
    }


def scan_dataset(dataset_root: Path, min_side: int, blur_var_threshold: float) -> dict:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    per_class_counts: Dict[str, int] = {}
    issues: List[ImageIssue] = []

    by_hash: Dict[str, List[str]] = {}

    for class_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()] , key=lambda p: p.name.casefold()):
        label = class_dir.name
        files = [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ]
        per_class_counts[label] = len(files)

        for p in files:
            try:
                stats = image_basic_stats(p)

                if stats["min_side"] < min_side:
                    issues.append(
                        ImageIssue(
                            path=str(p),
                            issue="ultra_small",
                            details={"min_side": stats["min_side"], "threshold": min_side},
                        )
                    )

                # Lower variance => blur-ish
                if stats["gray_var_proxy"] < blur_var_threshold:
                    issues.append(
                        ImageIssue(
                            path=str(p),
                            issue="blurry_or_low_detail",
                            details={"gray_var_proxy": stats["gray_var_proxy"], "threshold": blur_var_threshold},
                        )
                    )

            except Exception as e:
                issues.append(
                    ImageIssue(
                        path=str(p),
                        issue="corrupt_or_unreadable",
                        details={"error": str(e)},
                    )
                )
                continue

            # exact duplicate detection by sha256
            try:
                h = file_sha256(p)
                by_hash.setdefault(h, []).append(str(p))
            except Exception as e:
                issues.append(
                    ImageIssue(
                        path=str(p),
                        issue="hash_failed",
                        details={"error": str(e)},
                    )
                )

    duplicates = [paths for paths in by_hash.values() if len(paths) > 1]

    # Summarize duplicates as issue items
    duplicate_issues: List[dict] = []
    for group in duplicates:
        duplicate_issues.append({"count": len(group), "paths": group[:50]})

    report = {
        "dataset_root": str(dataset_root),
        "min_side_threshold": min_side,
        "blur_var_threshold": blur_var_threshold,
        "per_class_counts": per_class_counts,
        "num_total_images": sum(per_class_counts.values()),
        "num_issues": len(issues),
        "issues": [issue.__dict__ for issue in issues],
        "duplicate_groups": duplicate_issues,
    }

    return report


def write_report(report: dict, out_json: Path, out_txt: Path | None):
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if out_txt:
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        # write one path per line for images that have issues
        bad_paths = sorted({i["path"] for i in report.get("issues", [])})
        out_txt.write_text("\n".join(bad_paths), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Dataset preflight: duplicates + tiny/blurry/corrupt images")
    parser.add_argument("--dataset", type=Path, default=Path("CroppedData"))
    parser.add_argument("--min-side", type=int, default=64, help="min image side (width or height)")
    parser.add_argument("--blur-var-threshold", type=float, default=250.0, help="variance proxy threshold; lower => blurier")
    parser.add_argument("--out-json", type=Path, default=Path("analytics/dataset_preflight_report.json"))
    parser.add_argument("--out-txt", type=Path, default=Path("analytics/dataset_preflight_bad_paths.txt"))
    args = parser.parse_args()

    report = scan_dataset(args.dataset, args.min_side, args.blur_var_threshold)
    write_report(report, args.out_json, args.out_txt)

    print("Dataset preflight complete")
    print("Classes:")
    for k, v in report["per_class_counts"].items():
        print(f"  {k}: {v}")
    print(f"Issues: {report['num_issues']}")
    print(f"Duplicates groups: {len(report['duplicate_groups'])}")
    print(f"Report: {args.out_json}")


if __name__ == "__main__":
    main()

