"""Score the same rep with each pose model, and diff the result.

    PYTHONPATH=. .venv/bin/python scripts/score_pose.py \
        --video volleyball_dataset/videos/rep_0014.mp4

Keypoint steadiness is one question; whether it changes the number the tool
reports is another, and this is the one that decides whether swapping the pose
model is worth doing. Everything else is held constant - same ball track, same
passer, same contact frame - so any difference is the pose model alone.
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np

from core.pipeline import analyze_video
from core.vball_detector import VballNetDetector

ANGLES = ("knee_angle", "elbow_angle", "arm_torso_angle", "torso_angle")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path,
                        help="One clip, reported in full.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Instead, score this many reps from the dataset "
                             "and aggregate. One clip tells you the backends "
                             "differ; only a sample tells you whether they "
                             "differ the same way every time.")
    parser.add_argument("--dataset", type=Path, default=Path("volleyball_dataset"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backends", nargs="+", default=["mediapipe", "rtmpose-m"])
    args = parser.parse_args()
    if not args.video and not args.sample:
        parser.error("give either --video or --sample N")
    return args


def score_one(video, backends):
    """One rep through each backend, sharing a ball track so the contact frame
    and the passer are identical and the pose model is the only difference."""
    track = VballNetDetector().track(video)
    out = {}
    for backend in backends:
        reps = analyze_video(video, ball_detections=track,
                             pose=backend).report.get("reps", [])
        out[backend] = reps[0] if reps else None
    return out


def sweep(args):
    rows = list(csv.DictReader((args.dataset / "labels.csv").open()))
    random.seed(args.seed)
    chosen = random.sample(rows, min(args.sample, len(rows)))
    first, second = args.backends[0], args.backends[1]

    scores = {key: [] for key in ("overall", "stability", "integrity", "kinetic")}
    spreads, failures, disagreed = {}, [], 0
    for position, row in enumerate(chosen, start=1):
        clip = args.dataset / "videos" / row["filename"]
        print(f"[{position}/{len(chosen)}] {row['filename']} (label {row['quality']})",
              flush=True)
        reports = score_one(clip, args.backends)
        if any(rep is None for rep in reports.values()):
            failures.append(row["filename"])
            continue
        if len({rep["frame_center"] for rep in reports.values()}) > 1:
            failures.append(row["filename"] + " (contact frames differed)")
            continue
        for key in scores:
            scores[key].append(reports[second]["scores"][key]
                               - reports[first]["scores"][key])
        if abs(scores["overall"][-1]) > 5:
            disagreed += 1
        for key, value in reports[first]["measurements"].items():
            other = reports[second]["measurements"].get(key)
            if value is None or other is None:
                continue
            spreads.setdefault(key, []).append(abs(other - value))

    counted = len(scores["overall"])
    print(f"\n{counted} reps scored by both"
          + (f", {len(failures)} skipped: {failures}" if failures else ""))
    if not counted:
        return

    print(f"\n{second} minus {first}, per component:")
    print(f"  {'':<12} {'median':>8} {'mean':>8} {'worst':>8} "
          f"{'{} higher'.format(second):>16}")
    for key, values in scores.items():
        v = np.array(values)
        print(f"  {key:<12} {np.median(v):>8.1f} {v.mean():>8.1f} "
              f"{v[np.argmax(np.abs(v))]:>8.1f} {(v > 0).mean():>15.0%}")
    print(f"\n  overall differs by more than 5 points on "
          f"{disagreed}/{counted} reps")

    print(f"\nwhere the two models disagree, by measurement:")
    ranked = sorted(spreads.items(), key=lambda kv: -np.median(kv[1]))
    for key, values in ranked[:8]:
        print(f"  {key:<30} median gap {np.median(values):>7.2f}")


def main():
    args = parse_args()
    if args.sample:
        return sweep(args)
    print(f"scoring {args.video} with {', '.join(args.backends)} ...", flush=True)
    reports = score_one(args.video, args.backends)

    if any(rep is None for rep in reports.values()):
        missing = [name for name, rep in reports.items() if rep is None]
        print(f"\nno rep scored by: {', '.join(missing)}")
        return

    frames = {name: rep["frame_center"] for name, rep in reports.items()}
    print(f"\ncontact frame: {frames}"
          + ("   (identical, as intended)" if len(set(frames.values())) == 1
             else "   <- DIFFERENT, the comparison is not clean"))

    names = list(reports)
    width = max(len(name) for name in names) + 2
    print(f"\n{'':<28}" + "".join(f"{name:>{width + 6}}" for name in names) + "      diff")
    for key in ("overall", "stability", "integrity", "kinetic"):
        values = [reports[name]["scores"][key] for name in names]
        print(f"  {key:<26}" + "".join(f"{v:>{width + 6}}" for v in values)
              + f"{max(values) - min(values):>10}")

    print()
    measurements = reports[names[0]]["measurements"]
    for key in sorted(measurements):
        values = [reports[name]["measurements"].get(key) for name in names]
        if any(v is None for v in values):
            continue
        spread = max(values) - min(values)
        flag = "  <-" if (key in ANGLES and spread > 10) or (
            key not in ANGLES and spread > 0.3) else ""
        print(f"  {key:<26}" + "".join(f"{v:>{width + 6}.2f}" for v in values)
              + f"{spread:>10.2f}{flag}")

    print("\n  critiques")
    for name in names:
        print(f"    {name}: {len(reports[name]['critiques'])} raised")
        for line in reports[name]["critiques"][:4]:
            print(f"      - {line}")


if __name__ == "__main__":
    main()
