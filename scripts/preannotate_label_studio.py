"""Export YOLO detections as Label Studio preannotations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from urllib.parse import quote

import yaml
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO on images and create Label Studio import JSON."
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--classes",
        type=Path,
        default=Path("configs/classes.yaml"),
        help="Canonical class-ID registry.",
    )
    parser.add_argument(
        "--local-files-root",
        type=Path,
        default=Path("data/raw"),
        help="Label Studio local-file document root.",
    )
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--model-version")
    return parser.parse_args()


def load_class_names(path: Path) -> list[str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = config.get("names")
    if isinstance(names, list):
        class_names = names
    elif isinstance(names, dict):
        indexed_names = {int(class_id): name for class_id, name in names.items()}
        expected_ids = set(range(len(indexed_names)))
        if set(indexed_names) != expected_ids:
            raise ValueError(f"{path} class IDs must be contiguous from 0")
        class_names = [indexed_names[class_id] for class_id in range(len(indexed_names))]
    else:
        raise ValueError(f"{path} must contain a 'names' list or mapping")

    if not class_names or any(not isinstance(name, str) for name in class_names):
        raise ValueError(f"{path} contains invalid class names")
    return class_names


def find_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"Image directory does not exist: {directory}")
    images = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No supported images found in: {directory}")
    return images


def local_file_url(image: Path, local_files_root: Path) -> str:
    root = local_files_root.resolve()
    resolved_image = image.resolve()
    try:
        relative_path = resolved_image.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Image {resolved_image} is outside local-files root {root}"
        ) from error

    encoded_path = quote(relative_path.as_posix(), safe="/")
    return f"/data/local-files/?d={encoded_path}"


def as_percentage(value: float, dimension: int) -> float:
    return round(100.0 * value / dimension, 6)


def main() -> None:
    args = parse_args()
    images = find_images(args.images)
    class_names = load_class_names(args.classes)

    model = YOLO(str(args.weights))
    model_class_ids = {int(class_id) for class_id in model.names}
    expected_class_ids = set(range(len(class_names)))
    if model_class_ids != expected_class_ids:
        raise ValueError(
            "Checkpoint and canonical registry class IDs differ: "
            f"checkpoint={len(model_class_ids)}, registry={len(class_names)}"
        )

    results = model.predict(
        source=[str(path) for path in images],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        batch=args.batch,
        device=args.device,
        verbose=False,
    )

    model_version = (
        args.model_version
        or f"{args.weights.parent.parent.name}:{args.weights.name}"
    )
    tasks = []
    class_counts: Counter[str] = Counter()
    detection_scores: list[float] = []

    for image, result in zip(images, results, strict=True):
        height, width = result.orig_shape
        prediction_results = []

        for index, box in enumerate(result.boxes):
            class_id = int(box.cls.item())
            score = float(box.conf.item())
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            x1, x2 = sorted((max(0.0, x1), min(float(width), x2)))
            y1, y2 = sorted((max(0.0, y1), min(float(height), y2)))
            if x2 <= x1 or y2 <= y1:
                continue

            class_name = class_names[class_id]
            class_counts[class_name] += 1
            detection_scores.append(score)
            prediction_results.append(
                {
                    "id": f"box-{index:03d}",
                    "type": "rectanglelabels",
                    "from_name": "identity",
                    "to_name": "image",
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "score": round(score, 6),
                    "value": {
                        "x": as_percentage(x1, width),
                        "y": as_percentage(y1, height),
                        "width": as_percentage(x2 - x1, width),
                        "height": as_percentage(y2 - y1, height),
                        "rotation": 0,
                        "rectanglelabels": [class_name],
                    },
                }
            )

        prediction_score = (
            mean(item["score"] for item in prediction_results)
            if prediction_results
            else 0.0
        )
        tasks.append(
            {
                "data": {
                    "image": local_file_url(image, args.local_files_root),
                },
                "predictions": [
                    {
                        "model_version": model_version,
                        "score": round(prediction_score, 6),
                        "result": prediction_results,
                    }
                ],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(tasks, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    empty_tasks = sum(not task["predictions"][0]["result"] for task in tasks)
    print(f"Wrote {len(tasks)} tasks to {args.output}")
    print(
        f"Detections: {len(detection_scores)}; empty tasks: {empty_tasks}; "
        f"mean confidence: {mean(detection_scores):.3f}"
        if detection_scores
        else f"Detections: 0; empty tasks: {empty_tasks}"
    )
    if class_counts:
        print("Most frequent predictions:")
        for class_name, count in class_counts.most_common(10):
            print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()
