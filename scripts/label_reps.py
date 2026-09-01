"""Review the reps the pipeline proposes, fix the contact frame, label the pass.

You are correcting proposals, not labelling from scratch, which is a great deal
faster and gives you a second number for free: how often the detector's
proposed contact frame needed moving is a measurement of the rep detector.

Deliberately shows the raw clip with no scores on it. Labelling off the
annotated output would make the ground truth circular, since the thing being
validated would be on screen while you decide.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2

from core.pipeline import analyze_video


ROTATIONS = {
    "none": None,
    "cw": cv2.ROTATE_90_CLOCKWISE,
    "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180": cv2.ROTATE_180,
}
WINDOW_FRAMES = 90
DISPLAY_WIDTH = 960
FIELDS = ("rep_id", "video", "contact_frame", "label", "zone", "labeler", "notes")
# The scale coaches already use. Ordinal rather than binary, which carries more
# information per rep - and at 100 reps that matters.
LABELS = {
    ord("3"): "3",   # setter can set every option without moving much
    ord("2"): "2",   # setter has to move, middle probably off, out of system
    ord("1"): "1",   # shank or overpass, or the setter has to run to it
    ord("0"): "0",   # ace or no touch
}
ZONES = {ord(str(zone)): str(zone) for zone in range(1, 7)}
HELP = [
    "left/right +-1    a/d +-10    space set contact here",
    "3 perfect   2 out of system   1 shank/overpass   0 ace",
    "x exclude    n skip    q quit",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Label passing reps for the dataset.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--labels", default=Path("data/labels.csv"), type=Path)
    parser.add_argument("--labeler", required=True, help="Who is labelling. Two "
                        "people labelling a shared subset is how you get an "
                        "inter-rater agreement number.")
    parser.add_argument("--rotate", default="none", choices=sorted(ROTATIONS))
    parser.add_argument("--start-frame", default=0, type=int)
    parser.add_argument("--max-frames", default=None, type=int)
    parser.add_argument("--repropose", action="store_true",
                        help="Ignore the cached proposals and run the pipeline again.")
    return parser.parse_args()


def proposals_for(args):
    """Proposed contact frames, cached because the pipeline is slow."""
    cache = args.video.with_suffix(".proposals.json")
    if cache.exists() and not args.repropose:
        return json.loads(cache.read_text())

    print(f"Running the pipeline over {args.video} to propose reps...")
    analysis = analyze_video(
        args.video,
        rotate=ROTATIONS[args.rotate],
        start_frame=args.start_frame,
        max_frames=args.max_frames,
    )
    contacts = [
        args.start_frame + rep["frame_center"]
        for rep in analysis.report["reps"]
    ]
    cache.write_text(json.dumps(contacts))
    print(f"Proposed {len(contacts)} reps. Cached in {cache.name}.")
    return contacts


def load_window(capture, centre, rotation):
    """Frames either side of a proposal, preloaded so scrubbing is instant."""
    start = max(0, centre - WINDOW_FRAMES)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = {}
    for index in range(start, centre + WINDOW_FRAMES + 1):
        success, frame = capture.read()
        if not success:
            break
        if rotation is not None:
            frame = cv2.rotate(frame, rotation)
        scale = DISPLAY_WIDTH / frame.shape[1]
        frames[index] = cv2.resize(frame, None, fx=scale, fy=scale)
    return frames


def draw(frame, index, contact, proposed, position, total, prompt=None):
    canvas = frame.copy()
    banner = [
        f"rep {position}/{total}   frame {index}"
        f"   contact {contact} ({index - contact:+d})"
        f"   proposed {proposed}",
    ] + (prompt if prompt else HELP)
    for row, text in enumerate(banner):
        y = 26 + row * 26
        cv2.putText(canvas, text, (12, y), 0, 0.6, (0, 0, 0), 4)
        cv2.putText(canvas, text, (12, y), 0, 0.6, (255, 255, 255), 1)
    marker = (60, 220, 60) if index == contact else (200, 200, 200)
    cv2.circle(canvas, (canvas.shape[1] - 30, 30), 12, marker, -1)
    return canvas


def ask_zone(frame, index, contact, proposed, position, total):
    """Second keypress: which zone the passer received in.

    Asked separately because 1, 2 and 3 already mean pass quality. Space skips
    it, so a drill where the zone never changes costs one extra key per rep and
    can be left blank without holding up the labelling.
    """
    prompt = [
        "zone the PASSER received in:  1-6",
        "space to leave blank",
    ]
    while True:
        cv2.imshow("label reps", draw(
            frame, index, contact, proposed, position, total, prompt))
        key = cv2.waitKey(0) & 0xFF
        if key in ZONES:
            return ZONES[key]
        if key in (ord(" "), 13, 10):
            return ""
        if key == ord("q"):
            raise KeyboardInterrupt


def already_labelled(path):
    if not path.exists():
        return set()
    with path.open() as handle:
        return {row["rep_id"] for row in csv.DictReader(handle)}


def append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def label(args):
    contacts = proposals_for(args)
    if not contacts:
        print("No reps proposed. Nothing to label.")
        return

    done = already_labelled(args.labels)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    rotation = ROTATIONS[args.rotate]
    written = 0
    try:
        for position, proposed in enumerate(contacts, start=1):
            rep_id = f"{args.video.stem}_{proposed:06d}"
            if rep_id in done:
                print(f"skipping {rep_id}, already labelled")
                continue

            frames = load_window(capture, proposed, rotation)
            if not frames:
                continue

            index = proposed
            contact = proposed
            while True:
                frame = frames.get(index)
                if frame is None:
                    index = min(frames, key=lambda f: abs(f - index))
                    continue

                cv2.imshow("label reps", draw(
                    frame, index, contact, proposed, position, len(contacts)))
                key = cv2.waitKey(0) & 0xFF

                if key in (81, ord("[")):
                    index -= 1
                elif key in (83, ord("]")):
                    index += 1
                elif key == ord("a"):
                    index -= 10
                elif key == ord("d"):
                    index += 10
                elif key == ord(" "):
                    contact = index
                elif key in LABELS or key == ord("x"):
                    # Excluded reps are written too, not silently dropped.
                    # Otherwise there is no record of how many were thrown out,
                    # and a rerun would offer them again.
                    label = LABELS.get(key, "excluded")
                    zone = "" if label == "excluded" else ask_zone(
                        frame, index, contact, proposed, position, len(contacts))
                    append_row(args.labels, {
                        "rep_id": rep_id,
                        "video": str(args.video),
                        "contact_frame": contact,
                        "label": label,
                        "zone": zone,
                        "labeler": args.labeler,
                        "notes": "",
                    })
                    written += 1
                    break
                elif key == ord("n"):
                    break
                elif key == ord("q"):
                    raise KeyboardInterrupt

                index = max(min(frames), min(max(frames), index))
    except KeyboardInterrupt:
        print("stopped")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f"Wrote {written} rows to {args.labels}")


if __name__ == "__main__":
    label(parse_args())
