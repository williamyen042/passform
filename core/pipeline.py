"""Run the PassForm pipeline over a video file and return a scored report.

Kept free of any rendering imports on purpose: the dataset build walks a folder
of clips through this module, and pulling in the PIL/OpenCV-GUI drawing path
would make that slower for no reason.
"""

from typing import List, NamedTuple, Optional

import cv2

from core.ball_tracker import track_ball, track_detections
from core.people import assign_roles, tracks_from_detector
from core.people_detector import PeopleDetector
from core.pose_extractor import PoseExtractor, square_crop, to_frame_coordinates
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


# Every pose the finer pass reads is a crop of one person, so MediaPipe is
# only ever asked for one. Finding the others is YOLO-pose's job.
SINGLE_POSE = 1


def analyze_video(
    video_path,
    ball_detector=None,
    people_detector=None,
    rotate=None,
    start_frame=0,
    max_frames=None,
):
    """Decode a clip, find the people, measure the passer, then score them.

    Two passes over the video. The first finds every person with YOLO-pose and
    the ball with the ball detector; only then is it known which person is
    passing. The second crops to that person and reads their pose at full
    detail with MediaPipe, which measures joints more precisely and more
    steadily than YOLO-pose does - elbow angles differ by about 14 degrees
    between the two, and MediaPipe's frame-to-frame jitter is half as large.

    rotate takes a cv2.ROTATE_* constant, for the phone footage that is
    written sideways with no orientation metadata OpenCV will act on.
    start_frame and max_frames exist because real sessions run for minutes and
    reprocessing all of it to look at one rally is a waste. Frame indices in
    the returned report are relative to start_frame.
    """
    people_detector = people_detector or PeopleDetector()

    people_per_frame = []
    ball_candidates = []
    fps = 30.0

    def first_pass(frame, frame_index):
        people_per_frame.append(people_detector.detect(frame, frame_index))
        ball_candidates.append(
            ball_detector.detect_candidates(frame, frame_index)
            if ball_detector is not None
            else []
        )

    fps = _walk(video_path, rotate, start_frame, max_frames, first_pass)
    frame_count = len(people_per_frame)

    passer, target = assign_roles(tracks_from_detector(people_per_frame))
    frames_landmarks = _measure_passer(
        video_path, rotate, start_frame, frame_count, passer,
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


def _walk(video_path, rotate, start_frame, max_frames, handle):
    """Run handle(frame, index) over the chosen span, returning the clip's fps."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    if start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    index = 0
    try:
        while max_frames is None or index < max_frames:
            success, frame = capture.read()
            if not success:
                break
            if rotate is not None:
                frame = cv2.rotate(frame, rotate)
            handle(frame, index)
            index += 1
    finally:
        capture.release()
    return fps


def _measure_passer(video_path, rotate, start_frame, frame_count, passer):
    """Second pass: read the passer's pose from a crop around them."""
    frames_landmarks = [None] * frame_count
    if passer is None:
        return frames_landmarks

    extractor = PoseExtractor(mode="video", num_poses=SINGLE_POSE)

    def measure(frame, index):
        box = passer.box(index)
        if box is None:
            return
        crop, placement = square_crop(frame, box)
        if crop.size == 0:
            return
        # Timestamps come from the frame index, not CAP_PROP_POS_MSEC: that
        # property is read after the decode so it reports the next frame, and
        # some codecs return 0 forever, which breaks MediaPipe's requirement
        # that timestamps strictly increase.
        result = extractor.process_frame(crop, int(index * 1000 / 30))
        landmarks = extractor.get_landmarks(result) if result.pose_landmarks else None
        if landmarks:
            frames_landmarks[index] = to_frame_coordinates(
                landmarks, placement, frame.shape,
            )

    _walk(video_path, rotate, start_frame, frame_count, measure)
    return frames_landmarks


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
