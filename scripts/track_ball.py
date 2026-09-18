"""Watch the perception layer work: the ball, and the people in frame.

    PYTHONPATH=. .venv/bin/python scripts/track_ball.py --video data/murphy1.mp4 \
        --people --out output/check.mp4

Prints where the ball was found and how steadily each person was followed, and
with --out writes a copy of the video with all of it drawn on. That video is
the point: a detection rate tells you how often something fired, not whether it
fired on the right thing, and the difference between those two is where every
wrong number in this project has come from so far.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from core.people import MIN_PERSON_FRAMES
from core.people_detector import PeopleDetector
from core.pipeline import _walk
from core.vball_detector import (DEFAULT_MODEL_PATH, VballNetDetector, flight,
                                 landing, predict, radius_pixels, touches)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None,
                        help="Write an annotated copy here to watch it.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Heatmap threshold. Lower finds more and invents more.")
    parser.add_argument("--trail", type=int, default=12,
                        help="Frames of path to draw behind the ball.")
    parser.add_argument("--people", action="store_true",
                        help="Also find and follow everyone in frame.")
    return parser.parse_args()


# One colour per tracking id, so an id switch shows up as a box changing colour
# rather than as a number you have to read off a moving box.
COLOURS = [(80, 220, 80), (255, 150, 60), (200, 100, 255), (60, 200, 255),
           (255, 90, 160), (120, 255, 220), (180, 180, 90), (100, 140, 255)]


def find_people(video_path):
    """Per-frame {tracking id: PersonFrame}, straight from the detector.

    Deliberately not core.people.tracks_from_detector: that drops the tracking
    id, and the id is exactly what has to be inspected here. A person whose id
    changes halfway through a rep is two people as far as everything
    downstream is concerned.
    """
    detector = PeopleDetector()
    per_frame = []
    _walk(video_path, None, 0, None,
          lambda frame, index: per_frame.append(detector.detect(frame, index) or {}))
    return per_frame


def summarise_people(per_frame, fps):
    counts = [len(people) for people in per_frame]
    spans = {}
    for index, people in enumerate(per_frame):
        for person_id in people:
            first, last, seen = spans.get(person_id, (index, index, 0))
            spans[person_id] = (min(first, index), max(last, index), seen + 1)

    print(f"\npeople in frame: median {int(np.median(counts))}, "
          f"min {min(counts)}, max {max(counts)}")
    print(f"  {len(spans)} tracking ids over {len(per_frame)} frames")
    short = 0
    for person_id, (first, last, seen) in sorted(spans.items(),
                                                 key=lambda kv: -kv[1][2]):
        covered = last - first + 1
        gaps = covered - seen
        if seen < MIN_PERSON_FRAMES:
            short += 1
            continue
        print(f"    id {person_id:>3}: seen {seen:4d} frames "
              f"({first / fps:5.2f}s - {last / fps:5.2f}s)"
              + (f", missing {gaps} inside that span" if gaps else ""))
    if short:
        print(f"    {short} more id(s) seen under {MIN_PERSON_FRAMES} frames - "
              f"dropped downstream, and usually a flicker rather than a person")


def summarise(track, fps, width, height):
    found = [d for d in track if d is not None]
    print(f"\nball found in {len(found)}/{len(track)} frames "
          f"({len(found) / max(len(track), 1):.0%})")
    if not found:
        print("  nothing tracked - wrong camera, or the threshold is too high")
        return
    peaks = [d.confidence for d in found]
    print(f"  heatmap peak: median {np.median(peaks):.2f}, min {min(peaks):.2f}")

    sizes = [radius_pixels(d, width) * 2 for d in found]
    print(f"  apparent ball size: median {np.median(sizes):.0f}px "
          f"(range {min(sizes):.0f}-{max(sizes):.0f}px as it moves toward and away)")

    speeds = flight(track, fps, width, height)
    if speeds:
        mph = np.array([v["mph"] for v in speeds.values()])
        print(f"  in-plane speed: median {np.median(mph):.0f} mph, "
              f"peak {mph.max():.0f} mph "
              f"(2D only - motion straight at the camera reads as zero)")

    hits = touches(track, fps, width, height)
    print(f"\n  ball changes direction (someone played it, or it hit the floor):")
    if not hits:
        print("    never - the track is too sparse, or nothing touched it")
    for index, turn, before, after in sorted(hits)[:12]:
        print(f"    {index / fps:6.2f}s   turn {turn:5.1f} deg   "
              f"{before:5.1f} mph in -> {after:5.1f} mph out")


def _hud(frame, index, fps, speeds, peak, last_touch, touched_at):
    """Big enough to read while the clip plays. That is the whole requirement."""
    panel = frame[:150, :470].copy()
    frame[:150, :470] = cv2.addWeighted(
        panel, 0.35, np.zeros_like(panel), 0.65, 0)
    now = speeds.get(index, {}).get("mph") if speeds else None
    cv2.putText(frame, f"{now:.0f} mph" if now else "-- mph",
                (16, 58), 0, 1.6, (0, 255, 255), 3)
    cv2.putText(frame, f"peak {peak:.0f}", (250, 58), 0, 1.0, (200, 200, 200), 2)
    cv2.putText(frame, f"{index / fps:6.2f}s", (16, 92), 0, 0.8, (255, 255, 255), 2)
    if last_touch is not None:
        cv2.putText(frame,
                    f"touch {touched_at / fps:.2f}s  {last_touch['turn']:.0f} deg  "
                    f"{last_touch['before']:.0f} -> {last_touch['after']:.0f} mph",
                    (16, 128), 0, 0.72, (0, 200, 255), 2)


def render(video_path, track, out_path, trail, people_per_frame=None,
           speeds=None, hit_frames=None, fps=30.0, video_track=None):
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    video_track = video_track if video_track is not None else track
    # The ball returning to the height it was played at is the moment somebody
    # has to handle it, so that is the height the landing marker reports.
    target_y = 0.6
    peak, last_touch, touched_at = 0.0, None, None
    index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        for back in range(min(trail, index + 1)):
            past = track[index - back] if index - back < len(track) else None
            if past is None:
                continue
            point = (int(past.center[0] * width), int(past.center[1] * height))
            fade = 1 - (back / max(trail, 1))
            cv2.circle(frame, point, 3, (0, int(200 * fade), int(255 * fade)), -1)
        if people_per_frame and index < len(people_per_frame):
            for person_id, person in people_per_frame[index].items():
                box = person.box
                colour = COLOURS[person_id % len(COLOURS)]
                p1 = (int(box[0] * width), int(box[1] * height))
                p2 = (int(box[2] * width), int(box[3] * height))
                cv2.rectangle(frame, p1, p2, colour, 2)
                cv2.putText(frame, f"#{person_id}", (p1[0], max(p1[1] - 8, 16)),
                            0, 0.7, colour, 2)
        here = track[index] if index < len(track) else None
        if here is not None:
            point = (int(here.center[0] * width), int(here.center[1] * height))
            cv2.circle(frame, point, 16, (0, 255, 255), 2)
            label = f"{here.confidence:.2f}"
            if speeds and index in speeds:
                label = f"{speeds[index]['mph']:.0f} mph"
            cv2.putText(frame, label, (point[0] + 20, point[1] - 12),
                        0, 0.7, (0, 255, 255), 2)
        # Where it is going: fitted from the last quarter second, drawn for the
        # next half. Beyond that the miss grows faster than the arc is worth.
        if here is not None:
            arc = predict(video_track, index, fps, width, height, horizon=0.5)
            if arc:
                for step, (_, ax, ay) in enumerate(arc):
                    if step % 3:
                        continue
                    fade = 1 - step / len(arc)
                    cv2.circle(frame, (int(ax * width), int(ay * height)),
                               max(2, int(5 * fade)), (120, 255, 120), -1)
                spot = landing(arc, target_y)
                if spot is not None:
                    _, lx, ly = spot
                    point = (int(lx * width), int(ly * height))
                    cv2.drawMarker(frame, point, (120, 255, 120),
                                   cv2.MARKER_TILTED_CROSS, 34, 3)
                    cv2.putText(frame, "predicted", (point[0] + 20, point[1] + 6),
                                0, 0.7, (120, 255, 120), 2)

        if hit_frames and index in hit_frames:
            last_touch = hit_frames[index]
            touched_at = index

        peak = max(peak, speeds[index]["mph"]) if speeds and index in speeds else peak
        _hud(frame, index, fps, speeds, peak, last_touch, touched_at)
        writer.write(frame)
        index += 1
    capture.release()
    writer.release()
    print(f"\nwrote {out_path} - watch it and check the marker is on the ball")


def main():
    args = parse_args()
    detector = VballNetDetector(model_path=args.model, threshold=args.threshold)
    capture = cv2.VideoCapture(str(args.video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    print(f"tracking {args.video} ...")
    started = time.time()
    track = detector.track(args.video)
    elapsed = time.time() - started
    print(f"{len(track)} frames in {elapsed:.1f}s "
          f"({len(track) / max(elapsed, 1e-6):.0f} fps)")
    summarise(track, fps, width, height)

    people_per_frame = None
    if args.people:
        print("\nfinding people ...")
        started = time.time()
        people_per_frame = find_people(args.video)
        print(f"{len(people_per_frame)} frames in {time.time() - started:.1f}s")
        summarise_people(people_per_frame, fps)

    if args.out:
        hits = touches(track, fps, width, height)
        render(args.video, track, args.out, args.trail, people_per_frame,
               flight(track, fps, width, height),
               {i: {"turn": turn, "before": before, "after": after}
                for i, turn, before, after in hits},
               fps=fps, video_track=track)


if __name__ == "__main__":
    main()
