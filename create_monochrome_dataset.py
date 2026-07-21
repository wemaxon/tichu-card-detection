from __future__ import annotations

import argparse
import sys
import shutil
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def convert_image(source: Path, destination: Path, single_channel: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source) as image:
        grayscale = image.convert("L")
        output = grayscale if single_channel else grayscale.convert("RGB")
        save_kwargs = {"quality": 95} if destination.suffix.lower() in {".jpg", ".jpeg"} else {}
        output.save(destination, **save_kwargs)


def copy_label(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def render_progress(current: int, total: int, label: str) -> None:
    if total <= 0:
        return

    width = 30
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    message = f"\r{label}: [{bar}] {current}/{total}"
    end = "\n" if current == total else ""
    print(message, end=end, file=sys.stdout, flush=True)


def write_dataset_yaml(source_yaml: Path, destination_yaml: Path, destination_root: Path) -> None:
    content = source_yaml.read_text(encoding="utf-8")
    lines = content.splitlines()

    for index, line in enumerate(lines):
        if line.startswith("path:"):
            lines[index] = f"path: {destination_root.resolve().as_posix()}"
            break

    destination_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dataset(source_root: Path, destination_root: Path, single_channel: bool, limit: int | None) -> tuple[int, int]:
    image_paths = [
        path
        for path in sorted(source_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and "images" in path.parts
    ]
    label_paths = [
        path
        for path in sorted(source_root.rglob("*"))
        if path.is_file() and path.suffix.lower() == ".txt" and "labels" in path.parts
    ]

    if limit is not None:
        image_paths = image_paths[:limit]
        image_stems = {path.stem for path in image_paths}
        label_paths = [path for path in label_paths if path.stem in image_stems]

    for index, path in enumerate(image_paths, start=1):
        relative_path = path.relative_to(source_root)
        destination_path = destination_root / relative_path
        convert_image(path, destination_path, single_channel=single_channel)
        render_progress(index, len(image_paths), "Converting images")

    for index, path in enumerate(label_paths, start=1):
        relative_path = path.relative_to(source_root)
        destination_path = destination_root / relative_path
        copy_label(path, destination_path)
        render_progress(index, len(label_paths), "Copying labels")

    return len(image_paths), len(label_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mirror a YOLO dataset into a monochrome copy. "
            "By default, images are saved as grayscale RGB so standard YOLO models still receive 3 channels."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/scenes"),
        help="Source dataset root. Default: data/scenes",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/scenes_monochrome"),
        help="Destination dataset root. Default: data/scenes_monochrome",
    )
    parser.add_argument(
        "--single-channel",
        action="store_true",
        help="Save true single-channel grayscale images instead of grayscale RGB.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only convert the first N images. Useful for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source.resolve()
    destination_root = args.destination.resolve()
    source_yaml = source_root / "scenes.yaml"

    if not source_root.is_dir():
        raise FileNotFoundError(f"Source dataset folder not found: {source_root}")

    if not source_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {source_yaml}")

    destination_root.mkdir(parents=True, exist_ok=True)
    image_count, label_count = build_dataset(
        source_root=source_root,
        destination_root=destination_root,
        single_channel=args.single_channel,
        limit=args.limit,
    )
    write_dataset_yaml(source_yaml, destination_root / "scenes.yaml", destination_root)

    print(f"Converted {image_count} image(s) to {destination_root}")
    print(f"Copied {label_count} label file(s)")
    print(f"Wrote dataset config: {destination_root / 'scenes.yaml'}")


if __name__ == "__main__":
    main()
