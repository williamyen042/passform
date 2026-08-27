import sys

import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from pprint import pprint

from core.angle_calculator import joint_angle, segment_angle_to_floor
from core.ball_detector import BallDetector
from core.ball_tracker import contact_frames, evaluate_fit, fit_segment, segments
from core.pipeline import analyze_video, rep_for_frame

VIDEO_PATH = "data/sample_video2.mp4"
OUTPUT_DIR = Path("output")
SAVE_OUTPUT_VIDEO = True
# How long the last frame stays up once playback ends, so the numbers can
# actually be read. Any key closes it sooner.
HOLD_SECONDS = 6
BALL_MODEL_PATH = "models/volleyball_ball/best.pt"
BALL_TARGET_CLASSES = ("ball", "sports ball")
BALL_CONFIDENCE = 0.15
BALL_IMAGE_SIZE = 1280

# The clip is letterboxed into a fixed box so the output size never depends on
# the input, which keeps portrait and landscape footage on the same layout.
VIDEO_BOX = (1280, 720)
PANEL_WIDTH = 400
STRIP_HEIGHT = 264
CANVAS_SIZE = (VIDEO_BOX[0] + PANEL_WIDTH, VIDEO_BOX[1] + STRIP_HEIGHT)
THUMBNAIL_SIZE = (344, 190)
PREDICTION_FRAMES = 12

# Colors are RGB. cv2 wants BGR, so anything handed to a cv2 call goes through
# bgr() rather than being written out twice in two orders.
INK = (231, 239, 237)
MUTED = (126, 151, 148)
ACCENT = (86, 196, 202)
PANEL_BG = (19, 28, 27)
STRIP_BG = (14, 22, 21)
LETTERBOX = (9, 14, 13)
RULE = (42, 55, 54)
GOOD = (110, 214, 166)
WARN = (217, 162, 72)
SKELETON = (220, 242, 239)
JOINT = (86, 196, 202)
BALL_BOX = (217, 162, 72)
TRAIL = (120, 214, 220)
PREDICTION = (110, 214, 166)
CONTACT_MARK = (217, 128, 128)

FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Futura.ttc",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
]
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]
DISPLAY_LANDMARKS = sorted({
    landmark_index
    for connection in POSE_CONNECTIONS
    for landmark_index in connection
})
LEFT = {"shoulder": 11, "elbow": 13, "wrist": 15, "hip": 23, "knee": 25, "ankle": 27}
RIGHT = {"shoulder": 12, "elbow": 14, "wrist": 16, "hip": 24, "knee": 26, "ankle": 28}

# Targets mirror the bands the scorer actually grades against, so a row reading
# "off" here always agrees with the score in the panel.
JOINT_ROWS = [
    ("Elbow angle", "elbow", (170, 180)),
    ("Knee angle", "knee", (120, 155)),
    ("Arm to torso", "arm_torso", (75, 115)),
    ("Torso lean", "torso", (50, 80)),
]


def load_font(size):
    for path in FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_HUGE = load_font(66)
FONT_BIG = load_font(34)
FONT_HEAD = load_font(20)
FONT_BODY = load_font(18)
FONT_LABEL = load_font(14)


def bgr(color):
    return (color[2], color[1], color[0])


class TextLayer:
    """Collects every string on a frame so PIL runs once instead of per call.

    Each draw_text used to convert the whole frame to PIL and back, which was
    affordable for a caption and is not for a panel full of numbers.
    """

    def __init__(self):
        self.items = []

    def add(self, text, position, font, color=INK, align="left"):
        self.items.append((str(text), position, font, color, align))

    def flush(self, canvas):
        if not self.items:
            return
        image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        for text, (x, y), font, color, align in self.items:
            if align == "right":
                x -= draw.textlength(text, font=font)
            draw.text((x, y), text, font=font, fill=color)
        canvas[:, :] = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        self.items.clear()


def wrap(text, font, width):
    image = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(image)
    lines = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if draw.textlength(candidate, font=font) <= width or not line:
            line = candidate
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def fit_video(frame):
    """Letterbox the clip into VIDEO_BOX and report how it was placed."""
    box_width, box_height = VIDEO_BOX
    height, width = frame.shape[:2]
    scale = min(box_width / width, box_height / height)
    resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
    canvas = np.full((box_height, box_width, 3), bgr(LETTERBOX), np.uint8)
    x = (box_width - resized.shape[1]) // 2
    y = (box_height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas, (x, y, resized.shape[1], resized.shape[0])


def to_pixels(point, placement):
    x, y, width, height = placement
    return int(x + point[0] * width), int(y + point[1] * height)


def draw_landmarks(video, landmarks, placement):
    points = [to_pixels((mark.x, mark.y), placement) for mark in landmarks]
    for start, end in POSE_CONNECTIONS:
        cv2.line(video, points[start], points[end], bgr(SKELETON), 2, cv2.LINE_AA)
    for index in DISPLAY_LANDMARKS:
        cv2.circle(video, points[index], 4, bgr(JOINT), -1, cv2.LINE_AA)


def draw_ball(video, detection, placement, text):
    if detection is None:
        return
    x1, y1, x2, y2 = detection.bbox
    left, top = to_pixels((x1, y1), placement)
    right, bottom = to_pixels((x2, y2), placement)
    pad = 4
    cv2.rectangle(video, (left - pad, top - pad), (right + pad, bottom + pad),
                  bgr(BALL_BOX), 2, cv2.LINE_AA)
    text.add(f"ball {detection.confidence:.2f}", (left - pad, top - pad - 20),
             FONT_LABEL, BALL_BOX)


def draw_trajectory(video, track, frame_index, placement):
    if track is None:
        return
    trail = [
        to_pixels(detection.center, placement)
        for detection in track.detections
        if detection.frame_index <= frame_index
    ]
    for index in range(1, len(trail)):
        weight = index / len(trail)
        cv2.line(video, trail[index - 1], trail[index], bgr(TRAIL),
                 1 + int(round(2 * weight)), cv2.LINE_AA)
    if trail:
        cv2.circle(video, trail[-1], 7, bgr(TRAIL), 2, cv2.LINE_AA)

    active = None
    for segment in segments(track):
        if segment.frame_indices[0] <= frame_index <= segment.frame_indices[-1]:
            active = segment
    fit = fit_segment(active)
    if fit is None:
        return
    projected = [
        to_pixels(evaluate_fit(fit, frame_index + step), placement)
        for step in range(PREDICTION_FRAMES)
    ]
    # Dashed, so a prediction never reads as something that was observed.
    for index in range(1, len(projected), 2):
        cv2.line(video, projected[index - 1], projected[index],
                 bgr(PREDICTION), 2, cv2.LINE_AA)


def draw_contacts(video, track, frame_index, contacts, placement):
    for detection in track.detections if track else []:
        if detection.frame_index in contacts and detection.frame_index <= frame_index:
            cv2.drawMarker(video, to_pixels(detection.center, placement),
                           bgr(CONTACT_MARK), cv2.MARKER_CROSS, 22, 2)


def pose_thumbnail(landmarks):
    """The athlete's shape at contact, drawn on its own so it stays readable."""
    width, height = THUMBNAIL_SIZE
    thumb = np.full((height, width, 3), bgr(PANEL_BG), np.uint8)
    if not landmarks:
        return thumb

    xs = [landmarks[i].x for i in DISPLAY_LANDMARKS]
    ys = [landmarks[i].y for i in DISPLAY_LANDMARKS]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
    scale = min(width, height) * 0.78 / span
    centre_x = (min(xs) + max(xs)) / 2
    centre_y = (min(ys) + max(ys)) / 2

    def place(mark):
        return (
            int(width / 2 + (mark.x - centre_x) * scale),
            int(height / 2 + (mark.y - centre_y) * scale),
        )

    points = [place(mark) for mark in landmarks]
    for start, end in POSE_CONNECTIONS:
        cv2.line(thumb, points[start], points[end], bgr(SKELETON), 2, cv2.LINE_AA)
    for index in DISPLAY_LANDMARKS:
        cv2.circle(thumb, points[index], 3, bgr(JOINT), -1, cv2.LINE_AA)
    return thumb


def joint_angles(landmarks):
    """Both sides at the contact frame, so asymmetry is visible."""
    if not landmarks:
        return {}
    angles = {}
    for name, side in (("left", LEFT), ("right", RIGHT)):
        angles[name] = {
            "elbow": joint_angle(landmarks[side["wrist"]], landmarks[side["elbow"]],
                                 landmarks[side["shoulder"]]),
            "knee": joint_angle(landmarks[side["hip"]], landmarks[side["knee"]],
                                landmarks[side["ankle"]]),
            "arm_torso": joint_angle(landmarks[side["hip"]], landmarks[side["shoulder"]],
                                     landmarks[side["wrist"]]),
            "torso": segment_angle_to_floor(landmarks[side["hip"]],
                                            landmarks[side["shoulder"]]),
        }
    return angles


def draw_panel(canvas, analysis, rep, text):
    x0 = VIDEO_BOX[0]
    canvas[:, x0:] = bgr(PANEL_BG)
    cv2.line(canvas, (x0, 0), (x0, CANVAS_SIZE[1]), bgr(RULE), 1)

    left = x0 + 28
    right = CANVAS_SIZE[0] - 28
    width = right - left

    text.add("PASSFORM", (left, 30), FONT_LABEL, ACCENT)

    if rep is None:
        text.add("Reading pose", (left, 62), FONT_BIG, INK)
        text.add("No rep in this part of the clip.", (left, 112), FONT_BODY, MUTED)
        return

    scores = rep["scores"]
    text.add("OVERALL FORM", (left, 66), FONT_LABEL, MUTED)
    text.add(scores["overall"], (left, 88), FONT_HUGE, INK)
    text.add(f"rep {rep['rep_index']} of {len(analysis.report['reps'])}",
             (left, 170), FONT_BODY, MUTED)

    y = 214
    for label, key in (("Stability", "stability"), ("Platform", "integrity"),
                       ("Kinetic", "kinetic")):
        value = scores[key]
        text.add(label.upper(), (left, y), FONT_LABEL, MUTED)
        text.add(value, (right, y - 6), FONT_HEAD, INK, align="right")
        bar_y = y + 24
        cv2.rectangle(canvas, (left, bar_y), (right, bar_y + 5), bgr(RULE), -1)
        filled = left + int(width * max(0, min(100, value)) / 100)
        color = GOOD if value >= 75 else (WARN if value >= 55 else CONTACT_MARK)
        cv2.rectangle(canvas, (left, bar_y), (filled, bar_y + 5), bgr(color), -1)
        y += 62

    cv2.line(canvas, (left, y + 6), (right, y + 6), bgr(RULE), 1)
    y += 30

    text.add("CONTACT", (left, y), FONT_LABEL, MUTED)
    text.add(f"frame {rep['frame_center']}", (left, y + 22), FONT_BIG, INK)
    source = "measured from ball arc" if rep["contact_source"] == "ball" else "pose estimate"
    text.add(source, (left, y + 68), FONT_BODY, MUTED)
    y += 106

    cv2.line(canvas, (left, y), (right, y), bgr(RULE), 1)
    y += 26
    text.add("WORK ON", (left, y), FONT_LABEL, MUTED)
    y += 24
    critique = rep["critiques"][0] if rep["critiques"] else "Nothing flagged."
    for line in wrap(critique, FONT_BODY, width)[:3]:
        text.add(line, (left, y), FONT_BODY, INK)
        y += 26

    y += 14
    text.add("BALL FLIGHT", (left, y), FONT_LABEL, MUTED)
    y += 24
    if analysis.ball_track is None:
        text.add("Not tracked in this gym", (left, y), FONT_BODY, MUTED)
    else:
        text.add(f"{len(analysis.ball_track)} frames tracked", (left, y), FONT_BODY, INK)

    # Pinned to the bottom of the panel rather than flowing after the text
    # above it, so it never moves when a critique wraps to another line.
    thumb_x = left
    thumb_y = CANVAS_SIZE[1] - 28 - THUMBNAIL_SIZE[1]
    text.add("POSE AT CONTACT", (thumb_x, thumb_y - 22), FONT_LABEL, ACCENT)
    return (thumb_x, thumb_y)


def draw_strip(canvas, rep, text):
    y0 = VIDEO_BOX[1]
    canvas[y0:, :VIDEO_BOX[0]] = bgr(STRIP_BG)
    cv2.line(canvas, (0, y0), (VIDEO_BOX[0], y0), bgr(RULE), 1)

    columns = (28, 330, 440, 560)
    text.add("JOINT ANGLES AT CONTACT", (columns[0], y0 + 16), FONT_LABEL, ACCENT)
    for label, x in zip(("LEFT", "RIGHT", "TARGET"), columns[1:]):
        text.add(label, (x, y0 + 16), FONT_LABEL, MUTED)

    angles = joint_angles(rep["landmarks"]) if rep else {}
    y = y0 + 50
    for label, key, (low, high) in JOINT_ROWS:
        text.add(label, (columns[0], y), FONT_BODY, INK)
        for side, x in (("left", columns[1]), ("right", columns[2])):
            value = angles.get(side, {}).get(key)
            if value is None or np.isnan(value):
                text.add("--", (x, y), FONT_BODY, MUTED)
                continue
            inside = low <= value <= high
            text.add(f"{value:.0f}\u00b0", (x, y), FONT_BODY, INK if inside else WARN)
        text.add(f"{low}\u2013{high}\u00b0", (columns[3], y), FONT_BODY, MUTED)
        y += 34

    # Platform shape sits apart because it is measured across both arms at
    # once, so a left/right split would be meaningless.
    panel_x = 820
    edge = VIDEO_BOX[0] - 30
    cv2.line(canvas, (panel_x - 44, y0 + 18), (panel_x - 44, CANVAS_SIZE[1] - 24),
             bgr(RULE), 1)
    text.add("PLATFORM SHAPE", (panel_x, y0 + 16), FONT_LABEL, ACCENT)
    text.add("TARGET", (edge, y0 + 16), FONT_LABEL, MUTED, align="right")

    y = y0 + 50
    y = _measure_rows(canvas, rep, text, panel_x, edge, y, (
        ("Wrist gap", "wrist_gap_ratio", (None, 0.65), "under 0.65", 2),
        ("Forearm spread", "forearm_parallel_delta", (None, 10), "under 10\u00b0", 0),
    ))

    # Centre of gravity is two thirds of the stability score and was visible
    # nowhere, which made a low Stability number impossible to act on.
    y += 18
    text.add("CENTRE OF GRAVITY", (panel_x, y), FONT_LABEL, ACCENT)
    y += 34
    _measure_rows(canvas, rep, text, panel_x, edge, y, (
        ("Hip depth", "cog_ratio", (0.32, 0.52), "0.32\u20130.52", 2),
        ("Weight offset", "balance_offset", (None, 0.22), "under 0.22", 2),
    ))


def _measure_rows(canvas, rep, text, x, edge, y, rows):
    """One row per measurement: name, value coloured by its band, and the band."""
    for label, key, (low, high), shown_target, places in rows:
        value = rep["measurements"].get(key) if rep else None
        text.add(label, (x, y), FONT_BODY, INK)
        if value is None:
            text.add("--", (x + 210, y), FONT_BODY, MUTED, align="right")
        else:
            inside = (low is None or value >= low) and value <= high
            shown = f"{value:.{places}f}" + ("\u00b0" if places == 0 else "")
            text.add(shown, (x + 210, y), FONT_BODY,
                     INK if inside else WARN, align="right")
        text.add(shown_target, (edge, y), FONT_BODY, MUTED, align="right")
        y += 34
    return y


def compose(frame, analysis, frame_index, rep, thumb, contacts):
    video, placement = fit_video(frame)
    text = TextLayer()

    landmarks = analysis.frames_landmarks[frame_index]
    if landmarks:
        draw_landmarks(video, landmarks, placement)
    draw_ball(video, analysis.ball_detections[frame_index], placement, text)
    draw_trajectory(video, analysis.ball_track, frame_index, placement)
    draw_contacts(video, analysis.ball_track, frame_index, contacts, placement)

    canvas = np.full((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), bgr(PANEL_BG), np.uint8)
    canvas[:VIDEO_BOX[1], :VIDEO_BOX[0]] = video

    thumb_at = draw_panel(canvas, analysis, rep, text)
    draw_strip(canvas, rep, text)
    if thumb_at is not None:
        tx, ty = thumb_at
        canvas[ty:ty + thumb.shape[0], tx:tx + thumb.shape[1]] = thumb
        cv2.rectangle(canvas, (tx, ty), (tx + thumb.shape[1], ty + thumb.shape[0]),
                      bgr(RULE), 1)
    text.add(f"FRAME {frame_index + 1} / {len(analysis.frames_landmarks)}",
             (VIDEO_BOX[0] - 24, VIDEO_BOX[1] - 34), FONT_LABEL, INK, align="right")
    text.flush(canvas)
    return canvas


def render(video_path, analysis, output_path=None):
    """Second pass: redraw the clip with the final, whole-video scores."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    # Each rep carries the pose at its own contact frame, drawn once rather
    # than rebuilt for every frame it is shown on.
    thumbnails = {}
    for rep in analysis.report["reps"]:
        rep["landmarks"] = analysis.frames_landmarks[rep["frame_center"]]
        thumbnails[rep["rep_index"]] = pose_thumbnail(rep["landmarks"])
    empty_thumb = pose_thumbnail(None)
    contacts = set(contact_frames(analysis.ball_track))

    writer = None
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 analysis.fps, CANVAS_SIZE)

    # Play at the clip's own speed rather than as fast as frames compose.
    delay = max(1, int(round(1000 / max(analysis.fps, 1))))
    quit_early = False

    try:
        for frame_index in range(len(analysis.frames_landmarks)):
            success, frame = capture.read()
            if not success:
                break

            rep = rep_for_frame(analysis.report, frame_index)
            thumb = thumbnails.get(rep["rep_index"], empty_thumb) if rep else empty_thumb
            canvas = compose(frame, analysis, frame_index, rep, thumb, contacts)

            if writer is not None:
                writer.write(canvas)
            cv2.imshow("PassForm", canvas)
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                quit_early = True
                break

        if not quit_early:
            print(f"Holding the last frame for {HOLD_SECONDS}s. Press any key to close.")
            cv2.waitKey(HOLD_SECONDS * 1000)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def main(video_path=None):
    video_path = Path(video_path or VIDEO_PATH)
    if not video_path.exists():
        raise SystemExit(f"No such video: {video_path}")

    ball_detector = BallDetector(
        model_path=BALL_MODEL_PATH,
        target_classes=BALL_TARGET_CLASSES,
        confidence=BALL_CONFIDENCE,
        image_size=BALL_IMAGE_SIZE,
    )
    print(f"Analysing {video_path} ...")
    analysis = analyze_video(video_path, ball_detector=ball_detector)

    pprint(analysis.report)
    # Named after the clip so analysing a second video does not quietly
    # overwrite the first one's output.
    output_path = OUTPUT_DIR / f"{video_path.stem}_scored.mp4"
    render(video_path, analysis, output_path=output_path if SAVE_OUTPUT_VIDEO else None)
    if SAVE_OUTPUT_VIDEO:
        print(f"Saved scored video to {output_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
