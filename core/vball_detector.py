"""Track the ball with VballNet, a motion model, instead of a per-frame detector.

YOLO looks at one frame and asks "what does this object look like?". On our
footage the answer is a 37px white smear, and yolov8x found it in 15% of frames
at a median confidence of 0.055 - close enough to noise that the tracker built
paths out of ceiling lights. This model is handed nine consecutive frames at
once and asks "what moved like a ball?", which is a question a volleyball
answers unambiguously. Measured on rep_0002: 48% of frames, one clean parabola,
and 132 fps on the CPU against 25 seconds a clip for the YOLO pass.

The model and its weights are not ours. Both come from VballNet, by
Alexander (github.com/asigatchov), released under the MIT licence:

    https://github.com/asigatchov/fast-volleyball-tracking-inference

The preprocessing and heatmap decoding below follow that project's
src/inference_onnx_seq_gray_v2.py: nine frames to grayscale at 512x288,
stacked as channels, and the output heatmaps thresholded, contoured, and
reduced to a centroid with image moments. The V4c checkpoint emits a radius
map after its heatmaps, which is where the ball's apparent size comes from,
and with it the only real-world scale anywhere in this project.

Fetch the weights with:

    mkdir -p models/vballnet && curl -sL -o \\
      models/vballnet/VballNetV4c_seq9_grayscale.onnx \\
      https://raw.githubusercontent.com/asigatchov/fast-volleyball-tracking-inference/master/models/VballNetV4c_seq9_grayscale_20260908_213829.onnx
"""

from pathlib import Path

import cv2
import numpy as np

from core.ball_detector import BallDetection

DEFAULT_MODEL_PATH = "models/vballnet/VballNetV4c_seq9_grayscale.onnx"
# Fixed by the checkpoint: nine grayscale frames at 512x288 in, nine heatmaps
# (plus nine radius planes) out.
SEQUENCE = 9
INPUT_WIDTH = 512
INPUT_HEIGHT = 288
DEFAULT_THRESHOLD = 0.5
# FIVB ball: 65-67cm around, so about 21cm across. This is the only real-world
# length in the frame we can count on, and because the model reports the ball's
# apparent size per frame, it measures the scale at the ball's own distance -
# no court homography, and perspective handled for free.
BALL_DIAMETER_M = 0.21
MPH_PER_MPS = 2.236936
# The radius map runs large. Measured against YOLO's boxes on 96 frames of
# murphy3 where both fired on the same ball: VballNet 30.6px, YOLO 25.9px, a
# ratio of 1.16. Since the ball's width is the ruler, an over-wide ball makes
# everything read slow, so the correction goes straight on the radius. Re-measure
# this on new footage - it is a property of the model, not a law of nature.
RADIUS_CALIBRATION = 1 / 1.16


class VballNetDetector:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, threshold=DEFAULT_THRESHOLD):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for VballNet tracking. "
                "Install project requirements first."
            ) from exc

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"VballNet weights not found at {path}. See the module docstring "
                "for the one-line download."
            )
        self.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.threshold = threshold

    def track(self, video_path, rotate=None):
        """One detection per frame of the clip, or None where the ball is hidden.

        Returns the same BallDetection the YOLO path produces, so everything
        downstream is unchanged - except that this comes out already ordered
        and already one ball per frame, so core.ball_tracker has nothing left
        to disambiguate.

        Decodes nine frames at a time rather than the whole file: at 1080p a
        frame is 6MB, so holding a six minute session in memory would ask for
        135GB. A window is 56MB.
        """
        detections = []
        for start, window in _windows(video_path, rotate):
            heatmaps = self.session.run(
                None, {self.input_name: _preprocess(window)},
            )[0]
            has_radius = heatmaps.shape[1] >= 2 * SEQUENCE
            for offset in range(len(window)):
                point = _decode_heatmap(heatmaps[0, offset], self.threshold)
                if point is None:
                    detections.append(None)
                    continue
                x, y, peak = point
                # V4c emits a radius map after its heatmaps, read at the
                # detected point. Carried in the bbox so the size survives in
                # the detection type the rest of the project already uses.
                radius = _NOMINAL_RADIUS
                if has_radius:
                    row = min(max(int(y * INPUT_HEIGHT), 0), INPUT_HEIGHT - 1)
                    column = min(max(int(x * INPUT_WIDTH), 0), INPUT_WIDTH - 1)
                    measured = float(heatmaps[0, SEQUENCE + offset, row, column])
                    if measured > 0:
                        radius = measured
                detections.append(BallDetection(
                    frame_index=start + offset,
                    center=(x, y),
                    bbox=(x - radius, y - radius, x + radius, y + radius),
                    confidence=peak,
                    class_name="ball",
                ))
        return detections


_NOMINAL_RADIUS = 0.008          # fraction of frame width, when the model has no radius map


def radius_pixels(detection, frame_width, calibration=RADIUS_CALIBRATION):
    """The ball's apparent radius in pixels, from the detection's box."""
    return (detection.bbox[2] - detection.bbox[0]) / 2 * frame_width * calibration


def predict(track, at_index, fps, frame_width, frame_height, history=0.25,
            horizon=1.0):
    """Where the ball is going, fitted from where it has just been.

    Horizontally a ball coasts, vertically it falls, so x is fitted straight
    and y as a parabola. Both are fitted in image space rather than in metres:
    the projection of a parabola is not quite a parabola, but over the second
    or so that matters here the difference is smaller than the detector's own
    jitter, and working in pixels avoids needing a depth we cannot measure.

    Returns (seconds ahead, x, y) in normalized coordinates, or None when there
    is not enough recent track to fit.
    """
    span = max(2, int(round(history * fps)))
    recent = [(i, d) for i, d in enumerate(track[:at_index + 1])
              if d is not None and i > at_index - span]
    if len(recent) < 4:
        return None
    times = np.array([(i - at_index) / fps for i, _ in recent])
    xs = np.array([d.center[0] for _, d in recent])
    ys = np.array([d.center[1] for _, d in recent])
    x_fit = np.polyfit(times, xs, 1)
    y_fit = np.polyfit(times, ys, 2)
    if y_fit[0] <= 0:
        # Curving upward in image space means rising ever faster, which no
        # ball does. The fit has locked onto noise.
        return None
    ahead = np.arange(1, int(round(horizon * fps)) + 1) / fps
    return [(float(t), float(np.polyval(x_fit, t)), float(np.polyval(y_fit, t)))
            for t in ahead]


def landing(arc, target_y):
    """First point on a predicted arc that falls back to a given height."""
    if not arc:
        return None
    for (t, x, y), (_, _, next_y) in zip(arc, arc[1:]):
        if y <= target_y <= next_y:
            return t, x, target_y
    return None


def flight(track, fps, frame_width, frame_height, smooth=3):
    """Per-frame ball speed in mph, and the direction it is travelling.

    Speed is measured in ball-diameters and converted with BALL_DIAMETER_M, so
    a ball far from the camera is not read as slow. What this cannot see is
    motion along the camera axis: a serve driven straight at the lens covers
    no pixels and reads as nearly stationary, so treat every number here as
    the in-plane component and a lower bound on the true speed.
    """
    seen = [(i, d) for i, d in enumerate(track) if d is not None]
    out = {}
    for position, (index, detection) in enumerate(seen):
        window = seen[max(0, position - smooth):position + smooth + 1]
        if len(window) < 2:
            continue
        first_index, first = window[0]
        last_index, last = window[-1]
        frames_apart = last_index - first_index
        if frames_apart <= 0:
            continue
        dx = (last.center[0] - first.center[0]) * frame_width
        dy = (last.center[1] - first.center[1]) * frame_height
        # Scale from the ball's size now, not from the ends of the window,
        # so the reading belongs to this frame's depth.
        radius = radius_pixels(detection, frame_width)
        if radius <= 0:
            continue
        metres_per_pixel = BALL_DIAMETER_M / (2 * radius)
        seconds = frames_apart / fps
        speed = np.hypot(dx, dy) * metres_per_pixel / seconds
        out[index] = {
            "mph": speed * MPH_PER_MPS,
            "mps": speed,
            "direction": (dx / max(np.hypot(dx, dy), 1e-9),
                          dy / max(np.hypot(dx, dy), 1e-9)),
            "radius_px": radius,
        }
    return out


def touches(track, fps, frame_width, frame_height, min_turn=40.0, window=4,
            min_mph=5.0):
    """Frames where the ball changed direction sharply - somebody played it.

    A ball in flight only accelerates downward, so its direction turns slowly
    and smoothly. A touch is the discontinuity. This needs no pose and no
    court: it is a property of the trajectory, which is why it holds up on
    footage where the passer is one of nine people in frame.

    Returns (frame_index, turn_degrees, mph_before, mph_after), strongest first.
    A bounce off the floor looks exactly like a pass here - telling those apart
    is a question about who was nearby, not about the ball.
    """
    seen = [(i, d) for i, d in enumerate(track) if d is not None]
    found = []
    for position in range(window, len(seen) - window):
        index, _ = seen[position]
        before = _leg(seen[position - window:position + 1], frame_width, frame_height, fps)
        after = _leg(seen[position:position + window + 1], frame_width, frame_height, fps)
        if before is None or after is None:
            continue
        turn = float(np.degrees(np.arccos(np.clip(
            np.dot(before["unit"], after["unit"]), -1.0, 1.0))))
        if turn < min_turn:
            continue
        # A ball moving at walking pace that appears to swerve is the centroid
        # of a thresholded blob wobbling, not a touch.
        if max(before["mph"], after["mph"]) < min_mph:
            continue
        found.append((index, turn, before["mph"], after["mph"]))

    # One touch produces a run of qualifying frames; keep the sharpest turn in
    # each run rather than reporting the same contact eight times.
    kept = []
    separation = max(1, int(round(0.25 * fps)))
    for entry in sorted(found, key=lambda item: -item[1]):
        if all(abs(entry[0] - other[0]) >= separation for other in kept):
            kept.append(entry)
    return [(_centre_of_gap(track, index), turn, before, after)
            for index, turn, before, after in kept]


def _centre_of_gap(track, index):
    """Put the contact inside the gap the ball leaves, not at the edge of it.

    While the ball is against a platform it is half hidden by hands and
    forearms and the detector loses it, so the last frame it was seen is
    always a little before the touch. On rep_0014 the ball was tracked
    descending to frame 131, disappeared for four frames, and came back rising
    at 136 - the contact is in there, not at 131, and 131 is what was being
    handed to the pose measurement.
    """
    following = next(
        (later for later in range(index + 1, min(index + 12, len(track)))
         if track[later] is not None),
        None,
    )
    if following is None or following - index <= 1:
        return index
    return (index + following) // 2


def pass_features(track, contact_frame, fps, frame_width, frame_height):
    """What the ball did through and after the pass.

    Every feature the project has had so far describes the passer's body, while
    the label describes where the ball ended up. These sit on the label's side
    of that gap: how hard it arrived, how steeply it left, how high it got, and
    how long it stayed up. Measured in ball diameters and degrees, so they mean
    the same thing near and far from the camera.

    Height and flight time are the pair a setter actually cares about - a ball
    that leaves at the right angle but never gets up is still unsettable.
    """
    seen = [(i, d) for i, d in enumerate(track) if d is not None]
    if not seen:
        return {}
    before = [(i, d) for i, d in seen if contact_frame - 0.25 * fps <= i <= contact_frame]
    after = [(i, d) for i, d in seen if contact_frame <= i <= contact_frame + 0.25 * fps]
    out = {
        "ball_in_mph": None, "ball_out_mph": None, "ball_in_angle": None,
        "ball_out_angle": None, "ball_turn": None, "pass_rise": None,
        "pass_hang_time": None, "pass_travel": None,
    }

    in_leg = _leg(before, frame_width, frame_height, fps) if len(before) >= 2 else None
    out_leg = _leg(after, frame_width, frame_height, fps) if len(after) >= 2 else None
    if in_leg:
        out["ball_in_mph"] = round(in_leg["mph"], 2)
        out["ball_in_angle"] = round(_elevation(in_leg["unit"]), 1)
    if out_leg:
        out["ball_out_mph"] = round(out_leg["mph"], 2)
        # Degrees above horizontal. A pass wants to go up; a shank goes flat or
        # sideways, and this is the number that says which happened.
        out["ball_out_angle"] = round(_elevation(out_leg["unit"]), 1)
    if in_leg and out_leg:
        out["ball_turn"] = round(float(np.degrees(np.arccos(np.clip(
            np.dot(in_leg["unit"], out_leg["unit"]), -1.0, 1.0)))), 1)

    contact = next((d for i, d in seen if i >= contact_frame), None)
    onward = [(i, d) for i, d in seen if i > contact_frame]
    if contact is not None and onward:
        ball_px = 2 * radius_pixels(contact, frame_width)
        apex_index, apex = min(onward, key=lambda pair: pair[1].center[1])
        rise_px = (contact.center[1] - apex.center[1]) * frame_height
        out["pass_rise"] = round(rise_px / max(ball_px, 1e-6), 2)
        out["pass_hang_time"] = round((apex_index - contact_frame) / fps, 3)
        travel_px = abs(apex.center[0] - contact.center[0]) * frame_width
        out["pass_travel"] = round(travel_px / max(ball_px, 1e-6), 2)
    return out


def _elevation(unit):
    """Degrees above horizontal for a direction in image space (y grows down)."""
    return float(np.degrees(np.arctan2(-unit[1], abs(unit[0]))))


def _leg(segment, frame_width, frame_height, fps):
    """Average velocity over a run of detections, in pixels and mph."""
    if len(segment) < 2:
        return None
    (first_index, first), (last_index, last) = segment[0], segment[-1]
    frames_apart = last_index - first_index
    if frames_apart <= 0:
        return None
    dx = (last.center[0] - first.center[0]) * frame_width
    dy = (last.center[1] - first.center[1]) * frame_height
    length = float(np.hypot(dx, dy))
    if length < 1e-6:
        return None
    radius = radius_pixels(first, frame_width)
    metres_per_pixel = BALL_DIAMETER_M / (2 * max(radius, 1e-6))
    speed = length * metres_per_pixel / (frames_apart / fps)
    return {"unit": np.array([dx / length, dy / length]), "mph": speed * MPH_PER_MPS}


def _windows(video_path, rotate):
    """(start_index, nine frames) until the video runs out.

    A trailing part-window is dropped rather than padded: the model was trained
    on nine real frames, and a window padded with copies makes the ball look
    stationary in exactly the way the model reads as "no ball".
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    start, window = 0, []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            window.append(cv2.rotate(frame, rotate) if rotate is not None else frame)
            if len(window) == SEQUENCE:
                yield start, window
                start += SEQUENCE
                window = []
    finally:
        capture.release()


def _preprocess(window):
    """Nine frames to one (1, 9, 288, 512) grayscale tensor."""
    stack = [
        cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (INPUT_WIDTH, INPUT_HEIGHT),
        ).astype(np.float32) / 255.0
        for frame in window
    ]
    return np.stack(stack)[None]


def _decode_heatmap(heatmap, threshold):
    """Centre of the strongest blob, normalized, or None when nothing fires."""
    _, binary = cv2.threshold(heatmap, threshold, 1.0, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        (binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    moments = cv2.moments(max(contours, key=cv2.contourArea))
    if moments["m00"] == 0:
        return None
    x = moments["m10"] / moments["m00"] / INPUT_WIDTH
    y = moments["m01"] / moments["m00"] / INPUT_HEIGHT
    return float(x), float(y), float(heatmap.max())
