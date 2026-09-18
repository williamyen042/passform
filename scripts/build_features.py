"""Turn each labelled clip into one row of features for the classifier.

Phase 1.3 of classifier_plan.md. The features are not new work: the scorer
already measures 17 numbers per rep, and its rule-based verdict comes along
for free as the Phase 2 baseline to beat.

    PYTHONPATH=. .venv/bin/python scripts/build_features.py

About 25 seconds a clip on this machine, so an hour-long dataset is a coffee
break. Rows are cached to features.jsonl as they finish and features.csv is
rendered from that, so an interrupted run picks up where it stopped.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from core.pipeline import analyze_video
import cv2

from core.vball_detector import VballNetDetector, pass_features

KEYS = ("rep_id", "quality", "position", "source_video", "filename",
        "duration", "candidates", "contact_source", "contact_frac", "baseline_hint",
        "score_overall", "score_stability", "score_integrity", "score_kinetic")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=Path("volleyball_dataset"))
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many clips. For a quick look.")
    parser.add_argument("--no-ball", action="store_true",
                        help="Skip VballNet and take contact from the pose "
                             "proxy, which is what produced the first, bad "
                             "feature set. Here to reproduce it, not to use.")
    return parser.parse_args()


def chosen_rep(reps, frame_count):
    """One clip is one rep, but the detector sometimes finds a second contact.

    Take the one nearest the middle of the clip: the annotator pressed S before
    the pass and the quality key after it, so the pass they judged is the one
    in the middle. A dig or a set that follows sits at the end.
    """
    if not reps:
        return None
    middle = frame_count / 2
    return min(reps, key=lambda rep: abs(rep["frame_center"] - middle))


def row_for(rep_row, clip, tracker=None):
    track = tracker.track(clip) if tracker is not None else None
    analysis = analyze_video(clip, ball_detections=track)
    capture = cv2.VideoCapture(str(clip))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    frames = len(analysis.frames_landmarks)
    rep = chosen_rep(analysis.report.get("reps", []), frames)
    row = {
        "rep_id": int(rep_row["rep_id"]),
        "quality": int(rep_row["quality"]),
        "position": int(rep_row["position"]),
        "source_video": rep_row["source_video"],
        "filename": rep_row["filename"],
        "duration": float(rep_row["duration"]),
        "candidates": len(analysis.report.get("reps", [])),
        "contact_source": None,
        "contact_frac": None,
        "baseline_hint": None,
        "score_overall": None, "score_stability": None,
        "score_integrity": None, "score_kinetic": None,
    }
    if rep is None:
        # Kept as a row rather than dropped, so "the pipeline found no contact
        # in N clips" is a number in the report instead of a silent shortfall.
        return row
    # What the ball did, alongside what the body did. The label is about the
    # ball, so these are the features on the label's side of the question.
    if track is not None:
        row.update(pass_features(
            track, rep["frame_center"], analysis.fps, width, height))
        # The path itself, kept in the cache but not in the CSV. A trajectory
        # model later will want the whole flight rather than eight summaries
        # of it, and re-deriving it means another 40 minutes of GPU. Stored as
        # [frame, x, y, radius], normalized, so it survives a re-encode.
        row["ball_path"] = [
            [index, round(d.center[0], 5), round(d.center[1], 5),
             round((d.bbox[2] - d.bbox[0]) / 2, 5)]
            for index, d in enumerate(track) if d is not None
        ]
    row.update(
        contact_source=rep["contact_source"],
        contact_frac=round(rep["frame_center"] / max(frames, 1), 3),
        baseline_hint=rep["form_pass_quality_hint"],
        score_overall=rep["scores"]["overall"],
        score_stability=rep["scores"]["stability"],
        score_integrity=rep["scores"]["integrity"],
        score_kinetic=rep["scores"]["kinetic"],
        **rep["measurements"],
    )
    return row


# Kept in features.jsonl, never in the CSV: one is a number per rep, the other
# is a few hundred points.
BULK_KEYS = {"ball_path"}


def render_csv(rows, path):
    """features.csv from the cache, same as labels.csv from metadata.json."""
    measured = sorted(
        set().union(*(row.keys() for row in rows)) - set(KEYS) - BULK_KEYS)
    fields = list(KEYS) + measured
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["rep_id"]):
            writer.writerow({k: v for k, v in row.items() if k not in BULK_KEYS})


def main():
    args = parse_args()
    labels = list(csv.DictReader((args.dataset / "labels.csv").open()))
    cache = args.dataset / "features.jsonl"
    done = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            row = json.loads(line)
            done[row["rep_id"]] = row

    tracker = None if args.no_ball else VballNetDetector()
    todo = [r for r in labels if int(r["rep_id"]) not in done][:args.limit]
    print(f"{len(labels)} labelled, {len(done)} already measured, {len(todo)} to go",
          flush=True)

    with cache.open("a") as handle:
        for index, rep_row in enumerate(todo, start=1):
            clip = args.dataset / "videos" / rep_row["filename"]
            started = time.time()
            try:
                row = row_for(rep_row, clip, tracker)
            except Exception as error:               # noqa: BLE001 - one bad
                print(f"  {clip.name}: {error}", file=sys.stderr, flush=True)
                continue                             # clip must not end the run
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            done[row["rep_id"]] = row
            print(f"  [{index}/{len(todo)}] {clip.name} label={row['quality']} "
                  f"candidates={row['candidates']} via={row['contact_source']} "
                  f"({time.time() - started:.1f}s)", flush=True)

    if done:
        render_csv(list(done.values()), args.dataset / "features.csv")
    found = sum(1 for row in done.values() if row["candidates"])
    print(f"\n{len(done)} rows, {found} with a detected contact, "
          f"{len(done) - found} without -> {args.dataset / 'features.csv'}")


if __name__ == "__main__":
    main()
