import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


DEFAULT_DATASET_DIR = Path("datasets/volleyball_ball")
DEFAULT_OUTPUT_DIR = Path("models/volleyball_ball")
DEFAULT_PROJECT_DIR = Path("runs/volleyball_ball")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 volleyball-ball detector for PassForm."
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        type=Path,
        help="Directory containing the YOLO dataset (images/, labels/, data.yaml).",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base YOLO model or checkpoint to fine-tune.",
    )
    parser.add_argument("--epochs", default=100, type=int)
    # Must match DEFAULT_IMAGE_SIZE in core/ball_detector.py.
    parser.add_argument("--imgsz", default=960, type=int)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--batch", default=8, type=int)
    parser.add_argument("--project", default=DEFAULT_PROJECT_DIR, type=Path)
    parser.add_argument("--name", default="train")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    return parser.parse_args()


def default_device():
    # Apple Silicon GPU. The first run of this script trained on CPU and took
    # three hours.
    return "mps" if torch.backends.mps.is_available() else "cpu"


def find_data_yaml(dataset_dir):
    candidates = sorted(dataset_dir.rglob("data.yaml"))
    if not candidates:
        candidates = sorted(dataset_dir.rglob("*.yaml"))
    if not candidates:
        raise FileNotFoundError(
            f"No YOLO data.yaml was found under {dataset_dir}."
        )
    return candidates[0]


def train(args):
    data_yaml = find_data_yaml(args.dataset_dir)
    model = YOLO(args.model)
    train_result = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )

    best_weights = find_best_weights(args, model, train_result)
    if not best_weights.exists():
        raise FileNotFoundError(f"Expected trained weights at {best_weights}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "best.pt"
    shutil.copy2(best_weights, destination)
    print(f"Saved PassForm ball model to {destination}")


def find_best_weights(args, model, train_result):
    candidate_dirs = [
        args.project / args.name,
        Path("runs") / "detect" / args.project / args.name,
    ]

    for source in (train_result, getattr(model, "trainer", None)):
        save_dir = getattr(source, "save_dir", None)
        if save_dir is not None:
            candidate_dirs.insert(0, Path(save_dir))

    for candidate_dir in candidate_dirs:
        best_weights = candidate_dir / "weights" / "best.pt"
        if best_weights.exists():
            return best_weights

    discovered = sorted(
        Path("runs").rglob("weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if discovered:
        return discovered[0]

    return args.project / args.name / "weights" / "best.pt"


if __name__ == "__main__":
    train(parse_args())
