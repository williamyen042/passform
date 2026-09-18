"""Run the PassForm pipeline over a video file and return a scored report.

Kept free of any rendering imports on purpose: the dataset build walks a folder
of clips through this module, and pulling in the PIL/OpenCV-GUI drawing path
would make that slower for no reason.
"""

from typing import List, NamedTuple, Optional

import cv2

from core.ball_tracker import track_ball, track_detections
from core.people import (arrival_from_ball, assign_roles, roles_from_ball,
                         target_displacement, tracks_from_detector)
from core.people_detector import PeopleDetector
from core.vball_detector import touches
from core.pose_extractor import PoseExtractor, square_crop, to_frame_coordinates
from core.scorer import analyze_frames


class VideoAnalysis(NamedTuple):
    # How far the setter had to travel to play the pass, in their own torso
    # lengths, and when they played it. LABELING.md defines the whole 0-3 scale
    # in these terms - "barely moves", "has to move", "has to run" - so this is
    # the rubric's own quantity rather than a proxy for it. None when the ball
    # never reaches anyone, which is itself what a 0 looks like.
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
    target_travel: Optional[float] = None
    arrival_frame: Optional[int] = None


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
    ball_detections=None,
    pose="mediapipe",
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
    pose picks which model reads the passer: "mediapipe" as before, or
    "rtmpose"/"rtmpose-s"/"rtmpose-m"/"rtmpose-x". Both return the same 33-slot
    layout, so everything downstream is unchanged and the two can be scored
    against each other on the same clip.

    ball_detections takes a whole-clip track computed elsewhere, which is how
    VballNet plugs in: it reads nine frames at a time, so it cannot answer the
    per-frame question ball_detector answers.

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

    if ball_detections is None:
        ball_track = track_ball(ball_candidates)
        ball_detections = track_detections(ball_track, frame_count)
    else:
        ball_track = None
        ball_detections = list(ball_detections)[:frame_count]
        ball_detections += [None] * (frame_count - len(ball_detections))

    tracks = tracks_from_detector(people_per_frame)
    passer, target = assign_roles(tracks)
    # When the ball is visible it decides who played it, because it is the only
    # witness that cannot be confused by someone standing with their hands
    # together. Platform shape stays as the fallback.
    contacts = None
    travel = arrival = None
    if ball_detections is not None:
        by_ball = roles_from_ball(
            tracks, ball_detections, fps, *_frame_size(video_path, rotate))
        if by_ball is not None:
            passer, target, contact_frame = by_ball
            # The same touch that chose the passer is the one that gets scored.
            arrival = arrival_from_ball(
                touches(ball_detections, fps, *_frame_size(video_path, rotate)),
                contact_frame)
            if arrival is not None:
                travel = target_displacement(target, contact_frame, arrival)
            contacts = [{
                "frame_index": contact_frame,
                "contact_source": "ball",
                "ball_contact_distance": None,
                "ball_confidence": getattr(
                    ball_detections[contact_frame], "confidence", None),
            }]
    # Only the second or so around contact is measured. Reading pose across
    # the whole clip spends most of its time on frames where the passer is
    # walking, occluded or half out of shot, and MediaPipe's output there is
    # visible as skeleton flicker on anyone watching the render.
    window = None
    if contacts:
        centre = contacts[0]["frame_index"]
        window = (max(0, centre - int(round(1.0 * fps))),
                  min(frame_count, centre + int(round(0.6 * fps))))
    frames_landmarks = _measure_passer(
        video_path, rotate, start_frame, frame_count, passer, fps, window, pose,
    )

    report = analyze_frames(
        frames_landmarks,
        fps=fps,
        ball_detections=ball_detections,
        contacts=contacts,
    )
    return VideoAnalysis(
        report,
        fps,
        frames_landmarks,
        ball_detections,
        ball_track,
        passer,
        target,
        travel,
        arrival,
    )


def _frame_size(video_path, rotate):
    capture = cv2.VideoCapture(str(video_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    # A rotated clip is measured in the orientation the rest of the pipeline
    # sees, not the one the container stores.
    if rotate in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        width, height = height, width
    return width, height


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


RTM_MODES = {"rtmpose": "performance", "rtmpose-s": "lightweight",
             "rtmpose-m": "balanced", "rtmpose-x": "performance"}


def _measure_passer(video_path, rotate, start_frame, frame_count, passer, fps,
                    window=None, pose="mediapipe"):
    """Second pass: read the passer's pose from a crop around them."""
    frames_landmarks = [None] * frame_count
    if passer is None:
        return frames_landmarks
    first, last = window if window else (0, frame_count)

    if pose in RTM_MODES:
        # RTMPose is top-down and takes the box directly, so there is no crop
        # and no crop-to-frame mapping - one fewer coordinate transform to be
        # wrong about.
        from core.rtm_extractor import RTMExtractor
        extractor = RTMExtractor(mode=RTM_MODES[pose])

        def measure_rtm(frame, index):
            if not first <= index < last:
                return
            box = passer.box(index)
            if box is None:
                return
            frames_landmarks[index] = extractor.landmarks_for_box(frame, box)

        _walk(video_path, rotate, start_frame, frame_count, measure_rtm)
        return frames_landmarks

    extractor = PoseExtractor(mode="video", num_poses=SINGLE_POSE)

    def measure(frame, index):
        if not first <= index < last:
            return
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
        result = extractor.process_frame(crop, int(index * 1000 / max(fps, 1)))
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
