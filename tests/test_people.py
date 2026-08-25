import unittest
from types import SimpleNamespace

from core.people import assign_roles, target_displacement, track_people


def landmark(x, y):
    return SimpleNamespace(x=x, y=y, visibility=1.0)


def person(x, platform=False):
    """A pose at horizontal position x, arms either at the sides or on a platform."""
    points = {
        0: (x, 0.18),
        11: (x - 0.05, 0.35), 12: (x + 0.05, 0.35),
        23: (x - 0.04, 0.56), 24: (x + 0.04, 0.56),
        25: (x - 0.05, 0.76), 26: (x + 0.05, 0.76),
        27: (x - 0.06, 0.95), 28: (x + 0.06, 0.95),
        29: (x - 0.06, 0.96), 30: (x + 0.06, 0.96),
        31: (x - 0.07, 0.96), 32: (x + 0.07, 0.96),
    }
    if platform:
        # Wrists together and held out in front of the torso.
        points.update({
            13: (x + 0.05, 0.44), 14: (x + 0.07, 0.44),
            15: (x + 0.19, 0.48), 16: (x + 0.21, 0.48),
        })
    else:
        # Arms hanging: still close together, still parallel, but not a platform.
        points.update({
            13: (x - 0.06, 0.46), 14: (x + 0.06, 0.46),
            15: (x - 0.06, 0.56), 16: (x + 0.06, 0.56),
        })

    pose = [landmark(x, 0.5) for _ in range(33)]
    for index, point in points.items():
        pose[index] = landmark(*point)
    return pose


class PeopleTest(unittest.TestCase):
    def test_two_people_stay_separate_tracks(self):
        frames = [[person(0.30), person(0.70)] for _ in range(40)]

        tracks = track_people(frames)

        self.assertEqual(len(tracks), 2)
        for track in tracks:
            self.assertEqual(len(track), 40)

    def test_passer_is_the_one_who_forms_a_platform(self):
        frames = []
        for index in range(40):
            passing = 18 <= index <= 24
            # Order swapped every frame, the way MediaPipe may return them.
            pair = [person(0.30, platform=passing), person(0.70)]
            frames.append(pair if index % 2 else list(reversed(pair)))

        passer, target = assign_roles(track_people(frames))

        self.assertIsNotNone(passer)
        self.assertIsNotNone(target)
        self.assertLess(passer.position(0)[0], 0.5)
        self.assertGreater(target.position(0)[0], 0.5)

    def test_idle_person_alone_is_still_returned_as_passer(self):
        frames = [[person(0.5)] for _ in range(30)]

        passer, target = assign_roles(track_people(frames))

        self.assertIsNotNone(passer)
        self.assertIsNone(target)

    def test_target_moving_before_contact_does_not_count_against_the_pass(self):
        # Target walks into position over the first 20 frames, then holds.
        frames = []
        for index in range(40):
            target_x = 0.55 + 0.005 * min(index, 20)
            frames.append([person(0.20, platform=index == 20), person(target_x)])

        passer, target = assign_roles(track_people(frames))
        displacement = target_displacement(target, contact_frame=20, arrival_frame=39)

        self.assertIsNotNone(displacement)
        self.assertLess(displacement, 0.1)

    def test_target_moving_after_contact_is_measured(self):
        frames = []
        for index in range(40):
            target_x = 0.55 + (0.01 * (index - 20) if index > 20 else 0.0)
            frames.append([person(0.20, platform=index == 20), person(target_x)])

        passer, target = assign_roles(track_people(frames))
        displacement = target_displacement(target, contact_frame=20, arrival_frame=39)

        self.assertIsNotNone(displacement)
        self.assertGreater(displacement, 0.5)


if __name__ == "__main__":
    unittest.main()
