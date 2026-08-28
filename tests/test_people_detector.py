import unittest

import numpy as np

from core.people_detector import COCO_TO_MEDIAPIPE, MEDIAPIPE_LANDMARKS, to_mediapipe
from core.scorer import LEFT_SIDE, RIGHT_SIDE, _has_full_pose


class CocoToMediapipeTest(unittest.TestCase):
    def points(self):
        # 17 COCO keypoints in pixels, all confidently seen.
        return np.arange(34, dtype=float).reshape(17, 2) * 10, np.full(17, 0.9)

    def test_pose_is_the_length_the_scorer_expects(self):
        pose = to_mediapipe(*self.points(), width=1000, height=500)

        self.assertEqual(len(pose), MEDIAPIPE_LANDMARKS)
        self.assertTrue(_has_full_pose(pose))

    def test_every_joint_the_scorer_measures_is_present(self):
        pose = to_mediapipe(*self.points(), width=1000, height=500)

        for side in (LEFT_SIDE, RIGHT_SIDE):
            for name, index in side.items():
                self.assertGreater(pose[index].visibility, 0.5,
                                   f"{name} came through invisible")

    def test_coordinates_are_normalized_to_the_frame(self):
        points, scores = self.points()
        pose = to_mediapipe(points, scores, width=1000, height=500)

        coco_nose_x, coco_nose_y = points[0]
        self.assertAlmostEqual(pose[0].x, coco_nose_x / 1000)
        self.assertAlmostEqual(pose[0].y, coco_nose_y / 500)

    def test_feet_stay_invisible_so_the_support_base_falls_back_to_ankles(self):
        pose = to_mediapipe(*self.points(), width=1000, height=500)

        # COCO has no heel or toe keypoints. _projected_balance_offset drops
        # anything under 0.35 visibility and reverts to the ankles alone.
        for index in (29, 30, 31, 32):
            self.assertLess(pose[index].visibility, 0.35)

    def test_mapping_targets_are_unique(self):
        targets = list(COCO_TO_MEDIAPIPE.values())

        self.assertEqual(len(targets), len(set(targets)))


if __name__ == "__main__":
    unittest.main()
