"""Run MediaPipe and RTMPose over the same rep and compare them.

    PYTHONPATH=. .venv/bin/python scripts/compare_pose.py \
        --video volleyball_dataset/videos/rep_0063.mp4 --out output/pose_compare.mp4

Both read the same passer, from the same box, on the same frames, so the only
variable is the pose model. The video is for judging whether the skeleton sits
on the body; the table is for the thing that actually decides this, which is
whether the joint angles are steady enough to measure a range from.

Jitter is the number to look at. Every kinetic measurement in the scorer is a
range over a window, and a range is set by the two most extreme frames, so a
model that drops a joint for one frame and puts it back writes that dropout
straight into the feature table. That is a real bug this project already hit:
an elbow reading 180, 118, 158 degrees across three consecutive frames.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from core import scorer
from core.angle_calculator import joint_angle
from core.people import tracks_from_detector
from core.people_detector import PeopleDetector
from core.pipeline import _walk
from core.pose_extractor import PoseExtractor, square_crop, to_frame_coordinates
from core.rtm_extractor import MODELS, RTMExtractor
from core.vball_detector import VballNetDetector, touches

JOINTS = ("elbow", "knee", "arm_torso")
MEDIAPIPE_COLOUR = (120, 200, 255)
RTM_COLOUR = (120, 255, 170)
BONES = [("shoulder", "elbow"), ("elbow", "wrist"), ("shoulder", "hip"),
         ("hip", "knee"), ("knee", "ankle")]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None,
                        help="Write a side-by-side render.")
    parser.add_argument("--mode", default="performance", choices=sorted(MODELS),
                        help="RTMPose size. performance is the 384x288 x-model.")
    return parser.parse_args()


def angles(landmarks, side):
    if landmarks is None or not scorer._has_full_pose(landmarks):
        return {name: float("nan") for name in JOINTS}
    return {
        "elbow": joint_angle(landmarks[side["wrist"]], landmarks[side["elbow"]],
                             landmarks[side["shoulder"]]),
        "knee": joint_angle(landmarks[side["hip"]], landmarks[side["knee"]],
                            landmarks[side["ankle"]]),
        "arm_torso": joint_angle(landmarks[side["hip"]], landmarks[side["shoulder"]],
                                 landmarks[side["wrist"]]),
    }


def steadiness(series):
    """Frame-to-frame change, and how often it is more than a joint can move.

    Thirty degrees in one frame at 60fps is 1800 deg/s at a single joint. Real
    movement does not do that; a tracker losing a keypoint does.
    """
    clean = np.array([v for v in series if not np.isnan(v)])
    if len(clean) < 3:
        return float("nan"), float("nan"), 0
    steps = np.abs(np.diff(clean))
    return float(np.median(steps)), float(steps.max()), int((steps > 30).sum())


def draw(frame, landmarks, side_map, colour, label, corner):
    height, width = frame.shape[:2]
    if landmarks is not None and scorer._has_full_pose(landmarks):
        for side in (scorer.LEFT_SIDE, scorer.RIGHT_SIDE):
            for start, end in BONES:
                a, b = landmarks[side[start]], landmarks[side[end]]
                if min(getattr(a, "visibility", 1), getattr(b, "visibility", 1)) < 0.3:
                    continue
                cv2.line(frame, (int(a.x * width), int(a.y * height)),
                         (int(b.x * width), int(b.y * height)), colour, 2, cv2.LINE_AA)
                for point in (a, b):
                    cv2.circle(frame, (int(point.x * width), int(point.y * height)),
                               3, colour, -1, cv2.LINE_AA)
    cv2.putText(frame, label, corner, 0, 1.0, colour, 2, cv2.LINE_AA)
    return frame


def main():
    args = parse_args()
    print(f"finding the passer in {args.video} ...")
    track = VballNetDetector().track(args.video)
    detector = PeopleDetector()
    per_frame = []
    fps = _walk(args.video, None, 0, None,
                lambda f, i: per_frame.append(detector.detect(f, i) or {}))
    capture = cv2.VideoCapture(str(args.video))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    hits = touches(track, fps, width, height)
    if not hits:
        raise SystemExit("no contact found - nothing to compare around")
    contact = min(hit[0] for hit in hits)
    tracks = tracks_from_detector(per_frame)
    ball = next((track[i] for step in range(8)
                 for i in (contact - step, contact + step)
                 if 0 <= i < len(track) and track[i] is not None), None)
    from core.people import _nearest_to
    passer = _nearest_to(tracks, contact, ball.center) if ball else None
    if passer is None:
        raise SystemExit("could not identify the passer")

    first = max(0, contact - int(round(1.0 * fps)))
    last = min(len(per_frame), contact + int(round(0.6 * fps)))
    print(f"contact at frame {contact} ({contact / fps:.2f}s), "
          f"comparing frames {first}-{last}")

    mediapipe = PoseExtractor(mode="video", num_poses=1)
    rtm = RTMExtractor(mode=args.mode)
    rows = {"mediapipe": [], "rtmpose": []}
    spent = {"mediapipe": 0.0, "rtmpose": 0.0}
    frames_out = []

    def measure(frame, index):
        if not first <= index < last:
            return
        box = passer.box(index)
        if box is None:
            return

        started = time.time()
        crop, placement = square_crop(frame, box)
        result = (mediapipe.process_frame(crop, int(index * 1000 / max(fps, 1)))
                  if crop.size else None)
        mp_marks = (to_frame_coordinates(mediapipe.get_landmarks(result), placement,
                                         frame.shape)
                    if result is not None and result.pose_landmarks else None)
        spent["mediapipe"] += time.time() - started

        started = time.time()
        rtm_marks = rtm.landmarks_for_box(frame, box)
        spent["rtmpose"] += time.time() - started

        for name, marks in (("mediapipe", mp_marks), ("rtmpose", rtm_marks)):
            rows[name].append((index, angles(marks, scorer.LEFT_SIDE)))

        if args.out:
            left = draw(frame.copy(), mp_marks, None, MEDIAPIPE_COLOUR,
                        "MediaPipe", (20, 44))
            right = draw(frame.copy(), rtm_marks, None, RTM_COLOUR,
                         f"RTMPose ({args.mode})", (20, 44))
            mark = "   <- contact" if index == contact else ""
            panel = np.hstack([left, right])
            cv2.putText(panel, f"frame {index}{mark}", (20, height - 24), 0, 0.9,
                        (255, 255, 255), 2, cv2.LINE_AA)
            frames_out.append(cv2.resize(panel, (panel.shape[1] // 2,
                                                 panel.shape[0] // 2)))

    _walk(args.video, None, 0, None, measure)

    counted = len(rows["mediapipe"])
    print(f"\n{counted} frames measured\n")
    print(f"{'':<12} {'joint':<10} {'found':>7} {'median step':>12} "
          f"{'worst step':>11} {'>30 deg jumps':>14}")
    for name in ("mediapipe", "rtmpose"):
        for joint in JOINTS:
            series = [entry[joint] for _, entry in rows[name]]
            found = sum(1 for v in series if not np.isnan(v))
            median, worst, jumps = steadiness(series)
            print(f"{name if joint == JOINTS[0] else '':<12} {joint:<10} "
                  f"{found:>4}/{counted:<3} {median:>11.1f} {worst:>11.1f} "
                  f"{jumps:>14}")
    for name in ("mediapipe", "rtmpose"):
        print(f"  {name}: {spent[name]:.1f}s total, "
              f"{spent[name] / max(counted, 1) * 1000:.0f} ms/frame")

    at_contact = {}
    for name in ("mediapipe", "rtmpose"):
        at_contact[name] = next(
            (entry for index, entry in rows[name] if index == contact), None)
    if all(at_contact.values()):
        print(f"\nat the contact frame:")
        for joint in JOINTS:
            a, b = at_contact["mediapipe"][joint], at_contact["rtmpose"][joint]
            print(f"  {joint:<10} mediapipe {a:6.1f}   rtmpose {b:6.1f}   "
                  f"difference {abs(a - b):5.1f}")

    if args.out and frames_out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (frames_out[0].shape[1], frames_out[0].shape[0]))
        for frame in frames_out:
            writer.write(frame)
        writer.release()
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
