import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from pprint import pprint

from core.pose_extractor import PoseExtractor
from core.scorer import analyze_frames

VIDEO_PATH = "data/sample_video1.mp4"
OUTPUT_PATH = "output/passform_scored.mp4"
SAVE_OUTPUT_VIDEO = True
SCORE_UPDATE_INTERVAL = 5
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


def draw_score_overlay(frame, report):
    overlay = frame.copy()
    height, width = frame.shape[:2]
    panel_height = 128

    cv2.rectangle(overlay, (0, 0), (width, panel_height), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    if not report or not report["reps"]:
        draw_text(
            frame=frame,
            text="PassForm: collecting pose data...",
            position=(20, 34),
            font=FONT_LARGE,
            background=(18, 18, 18),
        )
        return

    latest_rep = report["reps"][-1]
    scores = latest_rep["scores"]
    metrics = [
        ("Overall", report["overall_score"], (255, 255, 255)),
        ("Stability", scores["stability"], (80, 220, 255)),
        ("Elbows", scores["integrity"], (80, 255, 140)),
        ("Kinetic", scores["kinetic"], (120, 160, 255)),
    ]

    draw_text(
        frame=frame,
        text="PassForm Rep",
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

    draw_text(
        frame,
        f"Contact frame {latest_rep['frame_center']}",
        (max(20, width - 260), 18),
        FONT_SMALL,
        color=(235, 235, 235),
    )

    critique = latest_rep["critiques"][0] if latest_rep["critiques"] else ""
    if critique:
        draw_text(frame, critique[:90], (20, 98), FONT_SMALL, color=(235, 235, 235))


def draw_metric_labels(frame, landmarks, report):
    if not report or not report["reps"] or not landmarks:
        return

    latest_rep = report["reps"][-1]
    measurements = latest_rep["measurements"]
    scores = latest_rep["scores"]
    side = SIDE_LANDMARKS[latest_rep.get("side_used", "left")]

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


cap = cv2.VideoCapture(VIDEO_PATH)
extractor = PoseExtractor(mode="video")
printed_detection = False
frames_landmarks = []
latest_report = None
writer = None

if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30

if SAVE_OUTPUT_VIDEO:
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    current_landmarks = None
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    result = extractor.process_frame(frame, timestamp_ms)

    if result.pose_landmarks:
        landmarks = extractor.get_landmarks(result)
        current_landmarks = landmarks
        frames_landmarks.append(landmarks)

        if not printed_detection:
            print(f"Detected {len(landmarks)} landmarks")
            print("First landmark:", landmarks[0])
            printed_detection = True

        draw_landmarks(frame, landmarks)
    else:
        frames_landmarks.append(None)

    if len(frames_landmarks) % SCORE_UPDATE_INTERVAL == 0:
        latest_report = analyze_frames(frames_landmarks, fps=fps)

    draw_score_overlay(frame, latest_report)
    draw_metric_labels(frame, current_landmarks, latest_report)

    if writer is not None:
        writer.write(frame)

    cv2.imshow("PassForm Pose Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()

report = analyze_frames(frames_landmarks, fps=fps)
pprint(report)

if SAVE_OUTPUT_VIDEO:
    print(f"Saved scored video to {OUTPUT_PATH}")
