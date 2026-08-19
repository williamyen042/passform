import unittest

from core.ball_detector import BallDetection, BallDetector


def detector_for_tests():
    detector = object.__new__(BallDetector)
    detector.min_box_area = 0.00002
    detector.max_box_area = 0.03
    detector.max_aspect_ratio = 4.0
    return detector


def detection(center, bbox=(0.45, 0.45, 0.55, 0.55), confidence=0.8):
    return BallDetection(
        frame_index=0,
        center=center,
        bbox=bbox,
        confidence=confidence,
        class_name="ball",
    )


class BallDetectorTest(unittest.TestCase):
    def test_geometry_filter_rejects_very_elongated_box(self):
        detector = detector_for_tests()
        elongated = detection(
            (0.5, 0.5),
            bbox=(0.20, 0.49, 0.80, 0.51),
        )

        self.assertFalse(detector._passes_geometry_filter(elongated))

    def test_geometry_filter_rejects_oversized_box(self):
        detector = detector_for_tests()
        oversized = detection((0.5, 0.5), bbox=(0.20, 0.20, 0.80, 0.80))

        self.assertFalse(detector._passes_geometry_filter(oversized))

    def test_geometry_filter_accepts_ball_shaped_box(self):
        detector = detector_for_tests()

        self.assertTrue(detector._passes_geometry_filter(detection((0.5, 0.5))))


if __name__ == "__main__":
    unittest.main()
