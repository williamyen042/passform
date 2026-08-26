"""Pull frames out of a clip, either to box in Roboflow or as hard negatives.

Two jobs, one loop. Positives go to a plain folder you upload for labelling;
negatives go straight into the YOLO dataset with empty label files, because a
frame with no ball needs no annotation work at all.
"""

import argparse
from pathlib import Path

import cv2


DEFAULT_DATASET_DIR = Path("datasets/volleyball_ball")
ROTATIONS = {
    "none": None,
    "cw": cv2.ROTATE_90_CLOCKWISE,
    "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180": cv2.ROTATE_180,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frames for ball labelling or as hard negatives."
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument(
        "--negatives",
        action="store_true",
        help=(
            "Write empty .txt labels alongside the frames and drop them into "
            "the dataset split. Use for clips with no ball in shot."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Where to write frames. Defaults to the dataset split for "
             "--negatives, otherwise frames/<video stem>/ for upload.",
    )
    parser.add_argument(
        "--rotate",
        default="none",
        choices=sorted(ROTATIONS),
        help="Phone clips are regularly stored sideways with no metadata "
             "OpenCV will act on. Check one frame before extracting hundreds.",
    )
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR, type=Path)
    parser.add_argument("--split", default="train", choices=("train", "valid", "test"))
    parser.add_argument("--start-frame", default=0, type=int)
    parser.add_argument("--end-frame", default=None, type=int)
    parser.add_argument(
        "--stride",
        default=5,
        type=int,
        help="Consecutive frames are near duplicates that cost training time "
             "and teach nothing, so sample rather than take everything.",
    )
    parser.add_argument("--prefix", default=None)
    return parser.parse_args()


def extract(args):
    if args.negatives:
        images_dir = args.output or args.dataset_dir / args.split / "images"
        labels_dir = args.dataset_dir / args.split / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
    else:
        images_dir = args.output or Path("frames") / args.video.stem
        labels_dir = None
    images_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix or ("hard_negative" if args.negatives else "frame")
    rotation = ROTATIONS[args.rotate]

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    if args.start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    saved = 0
    frame_index = args.start_frame
    while True:
        success, frame = capture.read()
        if not success:
            break
        if args.end_frame is not None and frame_index > args.end_frame:
            break

        if (frame_index - args.start_frame) % max(args.stride, 1) == 0:
            if rotation is not None:
                frame = cv2.rotate(frame, rotation)
            stem = f"{prefix}_{args.video.stem}_{frame_index:06d}"
            cv2.imwrite(str(images_dir / f"{stem}.jpg"), frame)
            if labels_dir is not None:
                (labels_dir / f"{stem}.txt").write_text("")
            saved += 1

        frame_index += 1

    capture.release()
    kind = "hard-negative" if args.negatives else "unlabelled"
    print(f"Wrote {saved} {kind} frames to {images_dir}")
    if not args.negatives:
        print("Upload that folder to Roboflow, box the ball, export YOLOv8.")


if __name__ == "__main__":
    extract(parse_args())
