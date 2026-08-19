import unittest

from core.ball_detector import BallDetection
from core.ball_tracker import (
    contact_frames,
    evaluate_fit,
    fit_segment,
    segments,
    track_ball,
)


def detection(frame_index, center, confidence=0.4):
    x, y = center
    return BallDetection(
        frame_index=frame_index,
        center=(x, y),
        bbox=(x - 0.01, y - 0.01, x + 0.01, y + 0.01),
        confidence=confidence,
        class_name="ball",
    )


def pass_arc(length=40, contact=20):
    """Ball falls in, platform reverses it, ball rises away. Gravity throughout."""
    centers = []
    for frame in range(length):
        time = frame - contact
        x = 0.20 + 0.015 * frame
        # Falling before contact, rising after, same downward acceleration.
        speed = -0.022 if time < 0 else 0.022
        y = 0.70 - abs(time) * abs(speed) + 0.0004 * time * time
        centers.append((x, min(max(y, 0.0), 1.0)))
    return centers


def lights(frame, count=4):
    """Static high-confidence false positives, exactly what the model gives us."""
    spots = [(0.85, 0.18), (0.89, 0.18), (0.70, 0.11), (0.78, 0.21)][:count]
    return [detection(frame, spot, confidence=0.78) for spot in spots]


class BallTrackerTest(unittest.TestCase):
    def test_arc_is_recovered_from_higher_confidence_static_lights(self):
        arc = pass_arc()
        candidates = [
            lights(frame) + [detection(frame, center, confidence=0.31)]
            for frame, center in enumerate(arc)
        ]

        track = track_ball(candidates)

        self.assertIsNotNone(track)
        self.assertGreaterEqual(len(track), 30)
        for center in track.centers:
            self.assertLess(center[1], 0.9)
            self.assertGreater(center[1], 0.1)
        # None of the tracked points may be one of the lights.
        for center in track.centers:
            self.assertGreater(abs(center[0] - 0.85), 0.02)

    def test_only_static_lights_yields_no_track(self):
        candidates = [lights(frame) for frame in range(60)]

        self.assertIsNone(track_ball(candidates))

    def test_contact_is_the_vertical_velocity_reversal(self):
        arc = pass_arc(length=40, contact=20)
        candidates = [[detection(frame, center)] for frame, center in enumerate(arc)]

        track = track_ball(candidates)
        contacts = contact_frames(track)

        self.assertEqual(len(contacts), 1)
        self.assertLess(abs(contacts[0] - 20), 3)

    def test_track_splits_into_one_segment_per_flight(self):
        arc = pass_arc(length=40, contact=20)
        candidates = [[detection(frame, center)] for frame, center in enumerate(arc)]

        pieces = segments(track_ball(candidates))

        self.assertEqual(len(pieces), 2)

    def test_fitted_arc_extrapolates_beyond_observed_frames(self):
        arc = pass_arc(length=40, contact=20)
        candidates = [[detection(frame, center)] for frame, center in enumerate(arc)]
        outgoing = segments(track_ball(candidates))[-1]

        fit = fit_segment(outgoing)
        last = outgoing.frame_indices[-1]
        here = evaluate_fit(fit, last)
        ahead = evaluate_fit(fit, last + 6)

        # Still travelling forward and still rising six frames later.
        self.assertGreater(ahead[0], here[0])
        self.assertLess(ahead[1], here[1])

    def test_gap_in_detections_does_not_break_the_track(self):
        arc = pass_arc()
        candidates = [[detection(frame, center)] for frame, center in enumerate(arc)]
        candidates[12] = []
        candidates[13] = []

        track = track_ball(candidates)

        self.assertIsNotNone(track)
        self.assertGreaterEqual(len(track), 35)


if __name__ == "__main__":
    unittest.main()
