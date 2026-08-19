import argparse
from pathlib import Path

import cv2


DEFAULT_DATASET_DIR = Path("datasets/volleyball_ball")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add no-ball hard-negative frames to the YOLO volleyball dataset."
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR, type=Path)
    parser.add_argument("--split", default="train", choices=("train", "valid", "test"))
    parser.add_argument("--start-frame", default=0, type=int)
    parser.add_argument("--end-frame", default=None, type=int)
    parser.add_argument("--stride", default=10, type=int)
    parser.add_argument("--prefix", default="hard_negative")
    return parser.parse_args()


def add_hard_negatives(args):
    images_dir = args.dataset_dir / args.split / "images"
    labels_dir = args.dataset_dir / args.split / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    saved = 0
    frame_index = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_index < args.start_frame:
            frame_index += 1
            continue
        if args.end_frame is not None and frame_index > args.end_frame:
            break

        should_save = (frame_index - args.start_frame) % max(args.stride, 1) == 0
        if should_save:
            stem = f"{args.prefix}_{args.video.stem}_{frame_index:06d}"
            image_path = images_dir / f"{stem}.jpg"
            label_path = labels_dir / f"{stem}.txt"
            cv2.imwrite(str(image_path), frame)
            label_path.write_text("")
            saved += 1

        frame_index += 1

    cap.release()
    print(f"Added {saved} hard-negative frames to {args.dataset_dir / args.split}")


if __name__ == "__main__":
    add_hard_negatives(parse_args())
