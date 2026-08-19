"""Run the PassForm pipeline over a video file and return a scored report.

Kept free of any rendering imports on purpose: the dataset build walks a folder
of clips through this module, and pulling in the PIL/OpenCV-GUI drawing path
would make that slower for no reason.
"""

from typing import List, NamedTuple, Optional

import cv2

from core.ball_tracker import track_ball, track_detections
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


def analyze_video(video_path, ball_detector=None):
    """Decode a clip, extract pose (and optionally the ball), then score it."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    extractor = PoseExtractor(mode="video")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    frames_landmarks = []
    ball_candidates = []

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame_index = len(frames_landmarks)
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
            frames_landmarks.append(
                extractor.get_landmarks(result) if result.pose_landmarks else None
            )
    finally:
        capture.release()

    # Only the tracked path reaches the scorer. Raw top-1 detections are
    # dominated by static clutter, and the tracker is what tells a ball from
    # a ceiling light.
    ball_track = track_ball(ball_candidates)
    ball_detections = track_detections(ball_track, len(frames_landmarks))

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
    )


def rep_for_frame(report, frame_index):
    """Return the rep being shown at a given frame, or None before the first."""
    current = None
    for rep in report.get("reps", []):
        if rep["frame_start"] <= frame_index:
            current = rep
    return current
