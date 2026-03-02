"""Compare two YOLO fire models on a folder of sample images.

This script runs both models on the same input folder and writes annotated images
for each model into a test-results directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}



REPO_ROOT = Path(__file__).resolve().parents[1]


def _candidate_paths(raw_path: Path) -> list[Path]:
    """Generate likely filesystem locations for a user-provided path."""
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
        return candidates

    candidates.append(raw_path)
    candidates.append(Path.cwd() / raw_path)
    candidates.append(REPO_ROOT / raw_path)
    candidates.append(Path(__file__).resolve().parent / raw_path)
    return candidates


def resolve_existing_path(raw_path: Path, kind: str) -> Path:
    checked: list[str] = []
    for candidate in _candidate_paths(raw_path):
        resolved = candidate.resolve()
        checked.append(str(resolved))
        if resolved.exists():
            return resolved

    searched = "\n  - ".join(["", *checked])
    raise FileNotFoundError(
        f"{kind} not found: {raw_path}\nSearched in:{searched}\n"
        "Tip: pass an absolute path if your file is outside the repository."
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run classification_model.pt and best_nano_111.pt on the same image folder "
            "and save annotated outputs for side-by-side comparison."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True, help="Folder containing sample images.")
    parser.add_argument(
        "--improved-weights",
        type=Path,
        default=Path("classification_model.pt"),
        help="Path to improved model weights (default: classification_model.pt)",
    )
    parser.add_argument(
        "--baseline-weights",
        type=Path,
        default=Path("best_nano_111.pt"),
        help="Path to original/baseline model weights (default: best_nano_111.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_results"),
        help="Root output directory for annotated comparisons.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", type=str, default="0", help="Device, e.g. 0 or cpu.")
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    args.source_dir = resolve_existing_path(args.source_dir, "Source image folder")
    if not args.source_dir.is_dir():
        raise NotADirectoryError(f"Source image folder is not a directory: {args.source_dir}")

    args.improved_weights = resolve_existing_path(args.improved_weights, "Improved weights")
    args.baseline_weights = resolve_existing_path(args.baseline_weights, "Baseline weights")


def list_images(source_dir: Path) -> list[Path]:
    images = [p for p in sorted(source_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        raise ValueError(f"No images found in {source_dir}. Supported extensions: {sorted(IMAGE_EXTENSIONS)}")
    return images


def model_label(weights_path: Path) -> str:
    return weights_path.stem.replace(" ", "_")


def run_model(
    model: YOLO,
    label: str,
    images: list[Path],
    output_dir: Path,
    conf: float,
    imgsz: int,
    device: str,
) -> dict[str, dict[str, float | int]]:
    per_image_summary: dict[str, dict[str, float | int]] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        result = model.predict(
            source=str(image_path),
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )[0]

        annotated = result.plot()
        output_name = f"{image_path.stem}__{label}{image_path.suffix}"
        output_path = output_dir / output_name
        cv2.imwrite(str(output_path), annotated)

        boxes = result.boxes
        confidences = boxes.conf.tolist() if boxes is not None and boxes.conf is not None else []
        per_image_summary[image_path.name] = {
            "detections": len(confidences),
            "mean_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
            "max_confidence": round(max(confidences), 4) if confidences else 0.0,
        }

    return per_image_summary


def main() -> None:
    args = parse_args()
    validate_inputs(args)
    images = list_images(args.source_dir)

    improved_label = model_label(args.improved_weights)
    baseline_label = model_label(args.baseline_weights)

    improved_model = YOLO(str(args.improved_weights))
    baseline_model = YOLO(str(args.baseline_weights))

    improved_output = args.output_dir / improved_label
    baseline_output = args.output_dir / baseline_label

    improved_summary = run_model(
        model=improved_model,
        label=improved_label,
        images=images,
        output_dir=improved_output,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )
    baseline_summary = run_model(
        model=baseline_model,
        label=baseline_label,
        images=images,
        output_dir=baseline_output,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )

    comparison_summary = {
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "models": {
            improved_label: {"weights": str(args.improved_weights), "results_dir": str(improved_output)},
            baseline_label: {"weights": str(args.baseline_weights), "results_dir": str(baseline_output)},
        },
        "per_image": {
            improved_label: improved_summary,
            baseline_label: baseline_summary,
        },
    }

    summary_path = args.output_dir / "comparison_summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(comparison_summary, indent=2), encoding="utf-8")

    print(f"Saved annotated images for {len(images)} inputs.")
    print(f"- Improved model outputs: {improved_output}")
    print(f"- Baseline model outputs: {baseline_output}")
    print(f"- Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
