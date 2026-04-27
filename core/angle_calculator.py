import math

import numpy as np


def point(landmark):
    """Return a normalized 2D point from a MediaPipe landmark-like object."""
    return np.array([landmark.x, landmark.y], dtype=float)


def midpoint(a, b):
    return (point(a) + point(b)) / 2.0


def distance(a, b):
    return float(np.linalg.norm(point(a) - point(b)))


def joint_angle(a, b, c):
    """Return the angle at point b in degrees for the triplet a-b-c."""
    ba = point(a) - point(b)
    bc = point(c) - point(b)

    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)
    if ba_norm == 0 or bc_norm == 0:
        return float("nan")

    cosine = np.dot(ba, bc) / (ba_norm * bc_norm)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def segment_angle_to_floor(a, b):
    """Return a segment's acute angle to the horizontal floor line."""
    start = point(a)
    end = point(b)
    delta = end - start
    if np.linalg.norm(delta) == 0:
        return float("nan")

    return float(np.degrees(math.atan2(abs(delta[1]), abs(delta[0]))))


def segment_heading(a, b):
    """Return a segment heading in degrees, useful for parallelism checks."""
    start = point(a)
    end = point(b)
    delta = end - start
    if np.linalg.norm(delta) == 0:
        return float("nan")

    return float(np.degrees(math.atan2(delta[1], delta[0])))


def angle_difference(a, b):
    """Return the smallest difference between two headings in degrees."""
    if math.isnan(a) or math.isnan(b):
        return float("nan")

    diff = abs((a - b + 180.0) % 360.0 - 180.0)
    return min(diff, 180.0 - diff)
