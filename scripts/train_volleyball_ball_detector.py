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
        default="yolov8s.pt",
        help=(
            "Checkpoint to fine-tune, or an architecture .yaml to build. "
            "Use yolov8-p2.yaml for the small-object head: it adds a stride-4 "
            "detection layer, and a volleyball is around 20-45 px at the "
            "inference size."
        ),
    )
    parser.add_argument(
        "--weights",
        default=None,
        help=(
            "Pretrained weights to transfer when --model is an architecture "
            ".yaml. Defaults to the checkpoint matching the model's scale. "
            "Ignored when --model is already a checkpoint."
        ),
    )
    parser.add_argument("--epochs", default=100, type=int)
    # Must match DEFAULT_IMAGE_SIZE in core/ball_detector.py. 1280 because the
    # ball measured 32 px in a 1920-wide gym frame and 70 px close up, which
    # lands at 21-47 px here.
    parser.add_argument("--imgsz", default=1280, type=int)
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


def build_model(model, weights=None):
    """Load a checkpoint, or build an architecture and transfer weights into it."""
    model = str(model)
    built = YOLO(model)
    if not model.endswith(".yaml"):
        return built

    # Only shape-matching layers are copied, so the scales have to agree.
    # yolov8-p2.yaml builds at nano scale, and pairing it with yolov8s.pt
    # transferred 45 of 437 tensors - effectively random init, which will not
    # train on a few hundred images. Scale-matched, the same pair transfers
    # 219. Deriving the default here means the mismatch cannot happen by hand.
    return built.load(weights or matching_weights(model))


def matching_weights(model_yaml):
    """yolov8s-p2.yaml -> yolov8s.pt. Unscaled configs build at nano."""
    stem = Path(model_yaml).stem.split("-")[0]
    return f"{stem}.pt" if stem[-1] in "nsmlx" else f"{stem}n.pt"


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
    model = build_model(args.model, args.weights)
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
