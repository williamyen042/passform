import cv2

from core.pose_extractor import PoseExtractor

VIDEO_PATH = "data/sample_video.mp4"
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


def draw_landmarks(frame, landmarks):
    height, width = frame.shape[:2]
    points = [
        (int(landmark.x * width), int(landmark.y * height))
        for landmark in landmarks
    ]

    for start, end in POSE_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

    for point in points:
        cv2.circle(frame, point, 4, (0, 0, 255), -1)

cap = cv2.VideoCapture(VIDEO_PATH)
extractor = PoseExtractor(mode="video")
printed_detection = False

if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    result = extractor.process_frame(frame, timestamp_ms)

    if result.pose_landmarks:
        landmarks = extractor.get_landmarks(result)

        if not printed_detection:
            print(f"Detected {len(landmarks)} landmarks")
            print("First landmark:", landmarks[0])
            printed_detection = True

        draw_landmarks(frame, landmarks)

    cv2.imshow("PassForm Pose Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
