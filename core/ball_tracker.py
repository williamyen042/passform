"""Link per-frame ball candidates into a physically plausible flight path.

Tracking-by-detection. The detector fires on anything small and bright — gym
ceiling lights very much included — so the filter that actually discriminates
is motion, not appearance. A light sits still and a ball follows a projectile
arc, so candidates are linked by a constant-velocity gate and the resulting
tracks are kept or dropped on how far they travel and how well they fit
y = y0 + vy*t + 0.5*a*t^2.

A pass is two arcs, not one: the ball falls in, the platform reverses it.
Acceleration stays constant through that, but vertical velocity flips sign, so
the track is split at each reversal and each segment is fitted separately. The
reversal frame is ball contact.

All positions are normalized to the frame (0-1 on each axis) and time is
measured in frames, so fitted acceleration is in normalized-height per frame
squared, not m/s^2.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# Normalized units per frame. Generous on purpose: the gate is only meant to
# stop absurd jumps, and track scoring does the real discrimination.
MAX_SPEED_PER_FRAME = 0.12
# How long a track may coast through an occlusion before it is closed.
MAX_GAP_FRAMES = 3
# A real pass arc spans at least half a second; measured ball tracks ran
# 27-97 frames, measured junk ran 7-20.
MIN_TRACK_FRAMES = 8
# A static false positive never travels this far; a real pass always does.
MIN_TRACK_DISPLACEMENT = 0.08
# RMS fit error in normalized units, above which a path is not projectile-like.
# Calibrated, not guessed: real ball tracks on in-domain footage measured
# 0.0026-0.0043 and gym-ceiling-light tracks measured 0.0152-0.0601, so this
# sits in the gap with roughly 2x margin either side. Re-measure once the
# detector is retrained on own-gym footage.
MAX_FIT_RESIDUAL = 0.008
# Minimum vertical velocity flip for a reversal to count as contact rather
# than detector jitter.
MIN_CONTACT_REVERSAL = 0.008
SMOOTHING_RADIUS = 1


@dataclass
class BallTrack:
    detections: List[object] = field(default_factory=list)

    def __len__(self):
        return len(self.detections)

    @property
    def frame_indices(self):
        return [detection.frame_index for detection in self.detections]

    @property
    def centers(self):
        return [detection.center for detection in self.detections]

    @property
    def last_frame(self):
        return self.detections[-1].frame_index

    def predict(self, frame_index):
        """Constant-velocity guess at where this ball should be next."""
        if len(self.detections) < 2:
            return self.detections[-1].center

        (previous_x, previous_y) = self.detections[-2].center
        (last_x, last_y) = self.detections[-1].center
        span = self.detections[-1].frame_index - self.detections[-2].frame_index
        if span <= 0:
            return self.detections[-1].center

        step = (frame_index - self.detections[-1].frame_index) / span
        return (
            last_x + (last_x - previous_x) * step,
            last_y + (last_y - previous_y) * step,
        )

    def displacement(self):
        """Diagonal of the box the path sweeps out."""
        if not self.detections:
            return 0.0
        xs = [center[0] for center in self.centers]
        ys = [center[1] for center in self.centers]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys))

    def mean_confidence(self):
        return float(np.mean([d.confidence for d in self.detections]))


def track_ball(candidates_per_frame):
    """Return the one track that looks like a ball in flight, or None."""
    return select_flight_track(build_tracks(candidates_per_frame))


def build_tracks(candidates_per_frame):
    """Link candidates across frames with a constant-velocity gate.

    ponytail: greedy nearest-neighbour, first track claims the candidate. The
    principled version is Hungarian assignment over a cost matrix; that only
    matters once two real balls are in frame at once, which a passing drill
    does not do.
    """
    closed = []
    open_tracks = []

    for frame_index, candidates in enumerate(candidates_per_frame):
        unmatched = list(candidates or [])
        still_open = []

        for track in open_tracks:
            gap = frame_index - track.last_frame
            if gap > MAX_GAP_FRAMES:
                closed.append(track)
                continue

            predicted = track.predict(frame_index)
            best = None
            best_distance = None
            for candidate in unmatched:
                separation = _distance(candidate.center, predicted)
                if separation > MAX_SPEED_PER_FRAME * max(gap, 1):
                    continue
                if best_distance is None or separation < best_distance:
                    best, best_distance = candidate, separation

            if best is not None:
                track.detections.append(best)
                unmatched.remove(best)
            still_open.append(track)

        open_tracks = still_open
        for candidate in unmatched:
            open_tracks.append(BallTrack([candidate]))

    return closed + open_tracks


def select_flight_track(tracks):
    """Drop anything static or non-projectile, then keep the longest survivor."""
    survivors = []
    for track in tracks:
        if len(track) < MIN_TRACK_FRAMES:
            continue
        if track.displacement() < MIN_TRACK_DISPLACEMENT:
            continue
        residual = fit_residual(track)
        if residual is None or residual > MAX_FIT_RESIDUAL:
            continue
        survivors.append(track)

    if not survivors:
        return None

    return max(
        survivors,
        key=lambda track: (len(track), track.displacement()),
    )


def contact_frames(track):
    """Frames where the ball stops falling and starts rising: platform contact."""
    if track is None or len(track) < 3:
        return []

    frame_indices = track.frame_indices
    ys = _smooth([center[1] for center in track.centers], SMOOTHING_RADIUS)

    contacts = []
    for index in range(1, len(ys) - 1):
        falling = ys[index] - ys[index - 1]
        rising = ys[index + 1] - ys[index]
        # Image y grows downward, so a local maximum is the lowest point of
        # the flight, where the platform sent the ball back up.
        if falling > 0 and rising < 0 and (falling - rising) >= MIN_CONTACT_REVERSAL:
            contacts.append(frame_indices[index])
    return contacts


def segments(track):
    """Split a track at each contact so every piece is a single free flight."""
    if track is None or not track.detections:
        return []

    boundaries = set(contact_frames(track))
    pieces = [[]]
    for detection in track.detections:
        pieces[-1].append(detection)
        if detection.frame_index in boundaries:
            pieces.append([detection])

    return [BallTrack(piece) for piece in pieces if len(piece) >= 3]


def fit_segment(segment):
    """Least-squares projectile fit: x linear in t, y quadratic in t."""
    if segment is None or len(segment) < 3:
        return None

    times = np.asarray(segment.frame_indices, dtype=float)
    origin = times[0]
    times = times - origin
    xs = np.array([center[0] for center in segment.centers], dtype=float)
    ys = np.array([center[1] for center in segment.centers], dtype=float)

    return {
        "origin_frame": float(origin),
        "x_coefficients": np.polyfit(times, xs, 1),
        "y_coefficients": np.polyfit(times, ys, 2),
    }


def evaluate_fit(fit, frame_index):
    """Where the fitted arc says the ball is at a given frame."""
    if fit is None:
        return None
    time = frame_index - fit["origin_frame"]
    return (
        float(np.polyval(fit["x_coefficients"], time)),
        float(np.polyval(fit["y_coefficients"], time)),
    )


def fit_residual(track):
    """RMS distance between the observed path and its fitted arcs."""
    pieces = segments(track) or ([track] if len(track) >= 3 else [])
    errors = []
    for piece in pieces:
        fit = fit_segment(piece)
        if fit is None:
            continue
        for detection in piece.detections:
            predicted = evaluate_fit(fit, detection.frame_index)
            errors.append(_distance(predicted, detection.center))

    if not errors:
        return None
    return float(np.sqrt(np.mean(np.square(errors))))


def track_detections(track, frame_count):
    """Per-frame list aligned to the clip, holding only tracked detections."""
    aligned = [None] * frame_count
    if track is None:
        return aligned
    for detection in track.detections:
        if 0 <= detection.frame_index < frame_count:
            aligned[detection.frame_index] = detection
    return aligned


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _smooth(values, radius):
    if radius <= 0:
        return list(values)
    smoothed = []
    for index in range(len(values)):
        left = max(0, index - radius)
        right = min(len(values), index + radius + 1)
        smoothed.append(float(np.mean(values[left:right])))
    return smoothed
