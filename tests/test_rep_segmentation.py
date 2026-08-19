import unittest
from types import SimpleNamespace

from core.scorer import analyze_frames


def landmark(x, y):
    return SimpleNamespace(x=x, y=y, visibility=1.0)


def pose_frame(depth=0.0):
    """A standing pose; `depth` sinks the hips/knees to mimic loading a pass."""
    landmarks = [landmark(0.5, 0.5) for _ in range(33)]
    points = {
        0: (0.50, 0.18),
        11: (0.40, 0.35 + depth), 12: (0.60, 0.35 + depth),
        13: (0.44, 0.50), 14: (0.56, 0.50),
        15: (0.48, 0.62), 16: (0.52, 0.62),
        23: (0.42, 0.56 + depth), 24: (0.58, 0.56 + depth),
        25: (0.38, 0.76), 26: (0.62, 0.76),
        27: (0.35, 0.95), 28: (0.65, 0.95),
        29: (0.34, 0.96), 30: (0.66, 0.96),
        31: (0.32, 0.96), 32: (0.68, 0.96),
    }
    for index, point in points.items():
        landmarks[index] = landmark(*point)
    return landmarks


def clip(depths):
    return [pose_frame(depth) for depth in depths]


def depths_with_dips(length, dip_centers, dip=0.06, width=4):
    """Flat baseline with a triangular hip dip at each named frame."""
    depths = [0.0] * length
    for center in dip_centers:
        for offset in range(-width, width + 1):
            index = center + offset
            if 0 <= index < length:
                falloff = (width - abs(offset)) / width
                depths[index] = max(depths[index], dip * falloff)
    return depths


class RepSegmentationTest(unittest.TestCase):
    def test_two_separated_dips_produce_two_reps(self):
        frames = clip(depths_with_dips(90, [20, 65]))

        reps = analyze_frames(frames, fps=30)["reps"]

        self.assertEqual(len(reps), 2)
        self.assertEqual([rep["rep_index"] for rep in reps], [1, 2])
        centers = [rep["frame_center"] for rep in reps]
        self.assertLess(abs(centers[0] - 20), 4)
        self.assertLess(abs(centers[1] - 65), 4)

    def test_single_dip_still_produces_one_rep(self):
        frames = clip(depths_with_dips(90, [45]))

        reps = analyze_frames(frames, fps=30)["reps"]

        self.assertEqual(len(reps), 1)

    def test_dips_closer_than_min_separation_collapse_to_one_rep(self):
        # 0.8s at 30fps is 24 frames; these are 12 apart.
        frames = clip(depths_with_dips(90, [40, 52]))

        reps = analyze_frames(frames, fps=30)["reps"]

        self.assertEqual(len(reps), 1)

    def test_flat_clip_never_returns_zero_reps(self):
        reps = analyze_frames(clip([0.0] * 60), fps=30)["reps"]

        self.assertEqual(len(reps), 1)


if __name__ == "__main__":
    unittest.main()
