import unittest
from types import SimpleNamespace

from core.scorer import analyze_frames


def landmark(x, y):
    return SimpleNamespace(x=x, y=y, visibility=1.0)


def pose_frame():
    landmarks = [landmark(0.5, 0.5) for _ in range(33)]
    points = {
        0: (0.50, 0.18),
        11: (0.40, 0.35),
        12: (0.60, 0.35),
        13: (0.44, 0.50),
        14: (0.56, 0.50),
        15: (0.48, 0.62),
        16: (0.52, 0.62),
        23: (0.42, 0.56),
        24: (0.58, 0.56),
        25: (0.38, 0.76),
        26: (0.62, 0.76),
        27: (0.35, 0.95),
        28: (0.65, 0.95),
        29: (0.34, 0.96),
        30: (0.66, 0.96),
        31: (0.32, 0.96),
        32: (0.68, 0.96),
    }
    for index, point in points.items():
        landmarks[index] = landmark(*point)
    return landmarks


def ball(frame_index, center, confidence=0.9):
    x, y = center
    return SimpleNamespace(
        frame_index=frame_index,
        center=center,
        bbox=(x - 0.02, y - 0.02, x + 0.02, y + 0.02),
        confidence=confidence,
        class_name="sports ball",
        track_id=None,
    )


CONTACT = 20
CLIP_FRAMES = 41


def approach_and_leave():
    """Ball falls toward the platform at y=0.62, then leaves again."""
    return [
        max(0.62 - 0.025 * abs(frame - CONTACT), 0.05)
        for frame in range(CLIP_FRAMES)
    ]


class BallContactTest(unittest.TestCase):
    def setUp(self):
        self.frames = [pose_frame() for _ in range(CLIP_FRAMES)]

    def test_clean_approach_contact_leave_selects_true_contact(self):
        y_values = approach_and_leave()
        detections = [
            ball(index, (0.50, y_value))
            for index, y_value in enumerate(y_values)
        ]

        report = analyze_frames(self.frames, fps=30, ball_detections=detections)
        rep = report["reps"][0]

        self.assertEqual(rep["contact_source"], "ball")
        self.assertEqual(rep["frame_center"], CONTACT)
        self.assertEqual(rep["ball_confidence"], 0.9)

    def test_short_missing_detection_gap_still_selects_nearby_contact(self):
        y_values = approach_and_leave()
        detections = [
            ball(index, (0.50, y_value))
            for index, y_value in enumerate(y_values)
        ]
        detections[CONTACT] = None

        report = analyze_frames(self.frames, fps=30, ball_detections=detections)
        rep = report["reps"][0]

        self.assertEqual(rep["contact_source"], "ball")
        self.assertIn(rep["frame_center"], {CONTACT - 1, CONTACT, CONTACT + 1})

    def test_far_noisy_ball_falls_back_to_pose_proxy(self):
        detections = [
            ball(index, (0.10, 0.10), confidence=0.95)
            for index in range(len(self.frames))
        ]

        report = analyze_frames(self.frames, fps=30, ball_detections=detections)
        rep = report["reps"][0]

        self.assertEqual(rep["contact_source"], "pose_proxy")
        self.assertIsNone(rep["ball_contact_distance"])

    def test_no_ball_detections_preserves_existing_scorer_behavior(self):
        old_report = analyze_frames(self.frames, fps=30)
        new_report = analyze_frames(
            self.frames,
            fps=30,
            ball_detections=[None] * len(self.frames),
        )

        self.assertEqual(old_report["overall_score"], new_report["overall_score"])
        self.assertEqual(
            old_report["reps"][0]["frame_center"],
            new_report["reps"][0]["frame_center"],
        )
        self.assertEqual(new_report["reps"][0]["contact_source"], "pose_proxy")


if __name__ == "__main__":
    unittest.main()
