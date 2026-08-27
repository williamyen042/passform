"""Run the PassForm pipeline over a video file and return a scored report.

Kept free of any rendering imports on purpose: the dataset build walks a folder
of clips through this module, and pulling in the PIL/OpenCV-GUI drawing path
would make that slower for no reason.
"""

from typing import List, NamedTuple, Optional

import cv2

from core.ball_tracker import track_ball, track_detections
from core.people import assign_roles, track_people
from core.pose_extractor import PoseExtractor
from core.scorer import analyze_frames


class VideoAnalysis(NamedTuple):
    report: dict
    fps: float
    # Per-frame and index-aligned, so frames_landmarks[i] and
    # ball_detections[i] both describe frame i. Either can hold None for a
    # frame where nothing was found.
    frames_landmarks: List[Optional[list]]
    ball_detections: List[Optional[object]]
    # The single flight path the tracker kept, or None when nothing in the
    # clip moved like a ball.
    ball_track: Optional[object] = None
    # Whole-clip tracks for the two people we care about. target is None when
    # only one person is in frame.
    passer: Optional[object] = None
    target: Optional[object] = None


# Opt in to 2 only when a second person is genuinely in frame. MediaPipe
# returns slightly different landmarks in multi-pose mode even when it still
# finds exactly one person, and on single-person clips that shifted detected
# contact by up to 31 frames.
DEFAULT_NUM_POSES = 1


def analyze_video(
    video_path,
    ball_detector=None,
    num_poses=DEFAULT_NUM_POSES,
    rotate=None,
    start_frame=0,
    max_frames=None,
):
    """Decode a clip, extract pose (and optionally the ball), then score it.

    rotate takes a cv2.ROTATE_* constant. Phone footage is regularly written
    sideways with no orientation metadata for OpenCV to act on, and a sideways
    athlete wrecks both pose landmarks and the vertical axis the ball tracker
    assumes gravity runs along.

    start_frame and max_frames exist because real sessions are minutes long and
    reprocessing all of it to look at one rally is a waste. Frame indices in
    the returned report are relative to start_frame.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    if start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    extractor = PoseExtractor(mode="video", num_poses=num_poses)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    poses_per_frame = []
    ball_candidates = []

    try:
        while True:
            if max_frames is not None and len(poses_per_frame) >= max_frames:
                break

            success, frame = capture.read()
            if not success:
                break

            if rotate is not None:
                frame = cv2.rotate(frame, rotate)

            frame_index = len(poses_per_frame)
            ball_candidates.append(
                ball_detector.detect_candidates(frame, frame_index)
                if ball_detector is not None
                else []
            )
            # Timestamps come from the frame index, not CAP_PROP_POS_MSEC:
            # that property is read after capture.read() so it reports the
            # *next* frame, and some codecs return 0 forever, which breaks
            # MediaPipe's strictly-increasing timestamp requirement.
            result = extractor.process_frame(frame, int(frame_index * 1000 / fps))
            poses_per_frame.append(extractor.get_all_landmarks(result))
    finally:
        capture.release()

    # Only the passer is scored. Without this, a bystander in the background
    # can become pose[0] for part of the clip and silently corrupt the rep.
    frame_count = len(poses_per_frame)
    passer, target = assign_roles(track_people(poses_per_frame))
    frames_landmarks = (
        passer.aligned(frame_count) if passer is not None else [None] * frame_count
    )

    # Only the tracked path reaches the scorer. Raw top-1 detections are
    # dominated by static clutter, and the tracker is what tells a ball from
    # a ceiling light.
    ball_track = track_ball(ball_candidates)
    ball_detections = track_detections(ball_track, frame_count)

    report = analyze_frames(
        frames_landmarks,
        fps=fps,
        ball_detections=ball_detections,
    )
    return VideoAnalysis(
        report,
        fps,
        frames_landmarks,
        ball_detections,
        ball_track,
        passer,
        target,
    )


def rep_for_frame(report, frame_index):
    """Return the rep to display at a frame, or None before the first contact.

    Keyed on the contact frame rather than the start of the measurement
    window. The window opens half a second before contact, so keying on that
    put a finished score on screen while the ball was still in the air, which
    reads as though the pass had been graded before it happened.
    """
    current = None
    for rep in report.get("reps", []):
        if rep["frame_center"] <= frame_index:
            current = rep
    return current
