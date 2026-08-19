import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from pprint import pprint

from core.ball_detector import BallDetector
from core.ball_tracker import contact_frames, evaluate_fit, fit_segment, segments
from core.pipeline import analyze_video, rep_for_frame

VIDEO_PATH = "data/sample_video4.mp4"
OUTPUT_PATH = "output/passform_scored.mp4"
SAVE_OUTPUT_VIDEO = True
TRAIL_COLOR = (80, 200, 255)
PREDICTION_COLOR = (120, 255, 180)
CONTACT_COLOR = (60, 60, 255)
PREDICTION_FRAMES = 12
BALL_MODEL_PATH = "models/volleyball_ball/best.pt"
BALL_TARGET_CLASSES = ("ball", "sports ball")
BALL_CONFIDENCE = 0.15
BALL_IMAGE_SIZE = 960
FONT_PATHS = [
    "/Library/Fonts/Calibri.ttf",
    "/System/Library/Fonts/Supplemental/Calibri.ttf",
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
SIDE_LANDMARKS = {
    "left": {
        "shoulder": 11,
        "elbow": 13,
        "wrist": 15,
        "hip": 23,
        "knee": 25,
        "ankle": 27,
    },
    "right": {
        "shoulder": 12,
        "elbow": 14,
        "wrist": 16,
        "hip": 24,
        "knee": 26,
        "ankle": 28,
    },
}


def load_font(size):
    for path in FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_LARGE = load_font(30)
FONT_MEDIUM = load_font(24)
FONT_SMALL = load_font(19)


def draw_text(frame, text, position, font, color=(255, 255, 255), background=None):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    x, y = position

    if background is not None:
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)
        padding = 6
        draw.rounded_rectangle(
            (left - padding, top - padding, right + padding, bottom + padding),
            radius=5,
            fill=background,
        )

    draw.text((x, y), text, font=font, fill=color)
    frame[:, :] = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def text_width(text, font):
    image = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(image)
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return right - left


def landmark_point(frame, landmark):
    height, width = frame.shape[:2]
    return int(landmark.x * width), int(landmark.y * height)


def normalized_point(frame, point):
    height, width = frame.shape[:2]
    return int(point[0] * width), int(point[1] * height)


def draw_landmarks(frame, landmarks):
    height, width = frame.shape[:2]
    points = [
        (int(landmark.x * width), int(landmark.y * height))
        for landmark in landmarks
    ]

    for start, end in POSE_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

    for landmark_index in DISPLAY_LANDMARKS:
        cv2.circle(frame, points[landmark_index], 4, (0, 0, 255), -1)


def draw_ball_detection(frame, detection):
    if detection is None:
        return

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = detection.bbox
    left = int(x1 * width)
    top = int(y1 * height)
    right = int(x2 * width)
    bottom = int(y2 * height)
    center = normalized_point(frame, detection.center)

    cv2.rectangle(frame, (left, top), (right, bottom), (0, 220, 255), 2)
    cv2.circle(frame, center, 5, (0, 220, 255), -1)
    draw_text(
        frame,
        f"ball {detection.confidence:.2f}",
        (left, max(14, top - 22)),
        FONT_SMALL,
        color=(0, 220, 255),
        background=(20, 20, 20),
    )


def draw_trajectory(frame, track, frame_index):
    """Draw the flight path so far, plus where the fitted arc says it goes."""
    if track is None:
        return

    height, width = frame.shape[:2]

    def to_pixels(point):
        return int(point[0] * width), int(point[1] * height)

    trail = [
        to_pixels(detection.center)
        for detection in track.detections
        if detection.frame_index <= frame_index
    ]
    for index in range(1, len(trail)):
        # Older points fade out so the current position reads clearly.
        weight = index / len(trail)
        thickness = 1 + int(round(2 * weight))
        cv2.line(frame, trail[index - 1], trail[index], TRAIL_COLOR, thickness)
    if trail:
        cv2.circle(frame, trail[-1], 7, TRAIL_COLOR, 2)

    active = None
    for segment in segments(track):
        if segment.frame_indices[0] <= frame_index <= segment.frame_indices[-1]:
            active = segment
    fit = fit_segment(active)
    if fit is None:
        return

    projected = [
        to_pixels(evaluate_fit(fit, frame_index + step))
        for step in range(0, PREDICTION_FRAMES)
    ]
    # Dashed, so the prediction never reads as observed data.
    for index in range(1, len(projected), 2):
        cv2.line(frame, projected[index - 1], projected[index], PREDICTION_COLOR, 2)


def draw_contacts(frame, track, frame_index, contacts):
    height, width = frame.shape[:2]
    for detection in track.detections if track else []:
        if detection.frame_index not in contacts:
            continue
        if detection.frame_index > frame_index:
            continue
        center = int(detection.center[0] * width), int(detection.center[1] * height)
        cv2.drawMarker(frame, center, CONTACT_COLOR, cv2.MARKER_CROSS, 22, 2)


def draw_score_overlay(frame, report, rep):
    overlay = frame.copy()
    height, width = frame.shape[:2]
    panel_height = 128

    cv2.rectangle(overlay, (0, 0), (width, panel_height), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    if rep is None:
        draw_text(
            frame=frame,
            text="PassForm: collecting pose data...",
            position=(20, 34),
            font=FONT_LARGE,
            background=(18, 18, 18),
        )
        return

    scores = rep["scores"]
    metrics = [
        ("Overall", report["overall_score"], (255, 255, 255)),
        ("Stability", scores["stability"], (80, 220, 255)),
        ("Elbows", scores["integrity"], (80, 255, 140)),
        ("Kinetic", scores["kinetic"], (120, 160, 255)),
    ]

    draw_text(
        frame=frame,
        text=f"PassForm Rep {rep['rep_index']}/{len(report['reps'])}",
        position=(20, 18),
        font=FONT_MEDIUM,
    )

    metric_texts = [
        (f"{label}: {score}", color)
        for label, score, color in metrics
    ]
    gap = 38
    total_width = sum(
        text_width(text, FONT_SMALL)
        for text, _ in metric_texts
    ) + gap * (len(metric_texts) - 1)
    x = max(20, int((width - total_width) / 2))
    for text, color in metric_texts:
        draw_text(frame, text, (x, 58), FONT_SMALL, color=color)
        x += text_width(text, FONT_SMALL) + gap

    contact_label = f"Contact frame {rep['frame_center']}"
    contact_source = rep.get("contact_source")
    if contact_source:
        contact_label = f"{contact_label} ({contact_source})"
    draw_text(
        frame,
        contact_label,
        (max(20, width - 330), 18),
        FONT_SMALL,
        color=(235, 235, 235),
    )

    critique = rep["critiques"][0] if rep["critiques"] else ""
    if critique:
        draw_text(frame, critique[:90], (20, 98), FONT_SMALL, color=(235, 235, 235))


def draw_metric_labels(frame, landmarks, rep):
    if rep is None or not landmarks:
        return

    measurements = rep["measurements"]
    scores = rep["scores"]
    side = SIDE_LANDMARKS[rep.get("side_used", "left")]

    label_specs = [
        (
            side["elbow"],
            f"Elbow {measurements['elbow_angle']} | Elbow {scores['integrity']}",
            (80, 255, 140),
            (16, -34),
        ),
        (
            side["knee"],
            f"Knee {measurements['knee_angle']} | Stability {scores['stability']}",
            (80, 220, 255),
            (16, -12),
        ),
        (
            side["shoulder"],
            f"Body sync {measurements['shoulder_hip_sync_error']} | Kinetic {scores['kinetic']}",
            (120, 160, 255),
            (16, -44),
        ),
    ]

    for landmark_index, text, color, offset in label_specs:
        x, y = landmark_point(frame, landmarks[landmark_index])
        draw_text(
            frame,
            text,
            (x + offset[0], y + offset[1]),
            FONT_SMALL,
            color=color,
            background=(20, 20, 20),
        )

    left_shoulder = landmark_point(frame, landmarks[11])
    right_hip = landmark_point(frame, landmarks[24])
    torso_x = int((left_shoulder[0] + right_hip[0]) / 2)
    torso_y = int((left_shoulder[1] + right_hip[1]) / 2)
    draw_text(
        frame,
        f"Torso {measurements['torso_angle']}",
        (torso_x + 12, torso_y - 12),
        FONT_SMALL,
        color=(255, 230, 120),
        background=(20, 20, 20),
    )


def render(video_path, analysis, output_path=None):
    """Second pass: redraw the clip with the final, whole-video scores."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    writer = None
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            analysis.fps,
            (width, height),
        )

    contacts = set(contact_frames(analysis.ball_track))

    try:
        for frame_index, landmarks in enumerate(analysis.frames_landmarks):
            success, frame = capture.read()
            if not success:
                break

            if landmarks:
                draw_landmarks(frame, landmarks)
            draw_ball_detection(frame, analysis.ball_detections[frame_index])
            draw_trajectory(frame, analysis.ball_track, frame_index)
            draw_contacts(frame, analysis.ball_track, frame_index, contacts)

            rep = rep_for_frame(analysis.report, frame_index)
            draw_score_overlay(frame, analysis.report, rep)
            draw_metric_labels(frame, landmarks, rep)

            if writer is not None:
                writer.write(frame)

            cv2.imshow("PassForm", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def main():
    ball_detector = BallDetector(
        model_path=BALL_MODEL_PATH,
        target_classes=BALL_TARGET_CLASSES,
        confidence=BALL_CONFIDENCE,
        image_size=BALL_IMAGE_SIZE,
    )
    analysis = analyze_video(VIDEO_PATH, ball_detector=ball_detector)

    pprint(analysis.report)
    render(
        VIDEO_PATH,
        analysis,
        output_path=OUTPUT_PATH if SAVE_OUTPUT_VIDEO else None,
    )
    if SAVE_OUTPUT_VIDEO:
        print(f"Saved scored video to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
