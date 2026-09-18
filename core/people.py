"""Follow every person in the frame and work out which one is passing.

MediaPipe returns the poses in a frame in no particular order, so pose[0] in
one frame is not necessarily the same human as pose[0] in the next. Everything
downstream assumes one continuous person, so poses are linked into tracks by
proximity first, and roles are assigned to whole tracks afterwards.

Role assignment leans on the platform: the passer is whoever forms one. A
setter waiting, or somebody wandering through the background, does not.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from core.scorer import (
    LEFT_SIDE,
    RIGHT_SIDE,
    platform_score,
    wrists_above_shoulders,
)


# Normalized units per frame. People walk; they do not teleport.
MAX_PERSON_STEP = 0.08
# Frames a person may go undetected before their track is closed.
MAX_PERSON_GAP = 8
MIN_PERSON_FRAMES = 5
# Hands this far above the shoulders means the target is playing the ball.
TARGET_PLAY_HAND_HEIGHT = 0.35
# Nothing that arrives later than this belongs to the pass we just measured.
MAX_FLIGHT_SECONDS = 2.5


@dataclass
class PersonTrack:
    frame_indices: List[int] = field(default_factory=list)
    poses: List[list] = field(default_factory=list)
    # Normalized xyxy per frame, present only when a detector supplied them.
    # The passer's boxes are what the finer pose pass crops to.
    boxes: List[tuple] = field(default_factory=list)

    def box(self, frame_index):
        for index, box in zip(self.frame_indices, self.boxes):
            if index == frame_index:
                return box
        return None

    def __len__(self):
        return len(self.poses)

    @property
    def last_frame(self):
        return self.frame_indices[-1]

    def platform_scores(self):
        scores = [platform_score(pose) for pose in self.poses]
        return [score for score in scores if not math.isnan(score)]

    def peak_platform_score(self):
        scores = self.platform_scores()
        return max(scores) if scores else float("nan")

    def platform_prominence(self):
        """Peak platform score above this person's own resting level.

        Peak alone barely separates a passer from a bystander: on a two-person
        clip the passer peaked at 67 and someone standing still scored 63,
        because arms hanging at your sides are also close together and also
        parallel. A passer's platform appears and disappears, so measuring the
        peak against their own median separated the same clip 11 to 1.
        """
        scores = self.platform_scores()
        if not scores:
            return float("nan")
        return float(max(scores) - np.median(scores))

    def aligned(self, frame_count):
        """Per-frame landmark list for the whole clip, None where unseen."""
        frames = [None] * frame_count
        for frame_index, pose in zip(self.frame_indices, self.poses):
            if 0 <= frame_index < frame_count:
                frames[frame_index] = pose
        return frames

    def position(self, frame_index):
        """Hip midpoint at a frame, or None if this person was not seen."""
        for index, pose in zip(self.frame_indices, self.poses):
            if index == frame_index:
                return _hip_midpoint(pose)
        return None


def tracks_from_detector(people_per_frame):
    """Build one track per tracking id, rather than re-deriving identity.

    YOLO-pose already follows people between frames, so the proximity linking
    below is only needed for pose sources that do not.
    """
    tracks = {}
    for people in people_per_frame:
        for person_id, person in (people or {}).items():
            track = tracks.setdefault(person_id, PersonTrack())
            track.frame_indices.append(person.frame_index)
            track.poses.append(person.pose)
            track.boxes.append(person.box)

    kept = [t for t in tracks.values() if len(t) >= MIN_PERSON_FRAMES]
    return sorted(kept, key=len, reverse=True)


def track_people(poses_per_frame):
    """Link per-frame poses into one track per person."""
    closed = []
    open_tracks = []

    for frame_index, poses in enumerate(poses_per_frame):
        unmatched = [pose for pose in (poses or []) if _hip_midpoint(pose)]
        still_open = []

        for track in open_tracks:
            gap = frame_index - track.last_frame
            if gap > MAX_PERSON_GAP:
                closed.append(track)
                continue

            anchor = _hip_midpoint(track.poses[-1])
            best = None
            best_distance = None
            for pose in unmatched:
                separation = _distance(_hip_midpoint(pose), anchor)
                if separation > MAX_PERSON_STEP * max(gap, 1):
                    continue
                if best_distance is None or separation < best_distance:
                    best, best_distance = pose, separation

            if best is not None:
                track.frame_indices.append(frame_index)
                track.poses.append(best)
                unmatched.remove(best)
            still_open.append(track)

        open_tracks = still_open
        for pose in unmatched:
            open_tracks.append(PersonTrack([frame_index], [pose]))

    tracks = [t for t in closed + open_tracks if len(t) >= MIN_PERSON_FRAMES]
    return sorted(tracks, key=len, reverse=True)


def assign_roles(tracks):
    """Return (passer, target). Passer is whoever forms and releases a platform.

    ponytail: platform shape only. Once the detector can see the ball in your
    own gym, the stronger signal is whichever person's wrists are nearest the
    ball at the contact frame — use that and keep this as the fallback for
    clips with no usable ball track.
    """
    if not tracks:
        return None, None

    scored = [
        (track.platform_prominence(), track)
        for track in tracks
        if not math.isnan(track.platform_prominence())
    ]
    if not scored:
        return tracks[0], None

    scored.sort(key=lambda item: item[0], reverse=True)
    passer = scored[0][1]
    others = [track for track in tracks if track is not passer]
    target = max(others, key=len) if others else None
    return passer, target


def roles_from_ball(tracks, ball_detections, fps, frame_width, frame_height):
    """Passer, target and contact frame, decided by where the ball went.

    Whoever the ball is touching when it changes direction is the person who
    played it, and whoever it reaches next is who they played it to. That is
    the definition, not a proxy for it.

    assign_roles has to guess from platform shape, which cannot tell a passer
    from someone standing with their hands together waiting their turn.
    Measured on eight clips of murphy footage it picked the wrong person in
    four. This needs the ball, so assign_roles stays as the fallback for clips
    where the detector sees nothing.

    Returns (passer, target, contact_frame) or None.
    """
    from core.vball_detector import touches

    hits = touches(ball_detections, fps, frame_width, frame_height)
    if not hits or not tracks:
        return None

    # The first touch in the clip is the pass. The annotator starts the clip
    # as the rep begins and the ball is already in flight, so whatever is
    # played first is what they were labelling; everything after it is the
    # set, the dig, or the ball hitting the floor.
    #
    # Ranking by turn angle picked the setter every time - a set reverses the
    # ball almost perfectly, near 180 degrees, where a pass turns maybe 130.
    # Ranking by incoming speed happened to agree with time order on rep_0063,
    # but time order is the rule that matches how the clips were cut.
    #
    # A serve struck inside the clip would be the first touch and would be
    # wrong. Nothing in the current footage does that, and the tell would be a
    # touch that accelerates the ball rather than absorbing it.
    contact_frame = min(hits, key=lambda hit: hit[0])[0]
    # The contact frame now sits inside the gap the ball leaves while it is
    # against the platform, so there is usually no detection on it. Take the
    # nearest sighting either side for the proximity test - the ball is at the
    # passer for all of them, which is the entire point of the gap.
    ball = _nearest_sighting(ball_detections, contact_frame)
    if ball is None:
        return None

    passer = _nearest_to(tracks, contact_frame, ball.center)
    if passer is None:
        return None

    # The next touch in time is the arrival. Whoever is there is the target,
    # and if nobody is, the pass did not reach anyone - which is a 0 or a 1,
    # and the caller should be able to see that rather than have it hidden.
    target = None
    later = sorted(index for index, _, _, _ in hits if index > contact_frame)
    if later:
        arrival = later[0]
        arriving_ball = _nearest_sighting(ball_detections, arrival)
        if arriving_ball is not None:
            target = _nearest_to(tracks, arrival, arriving_ball.center)
    return passer, target, contact_frame


# Beyond this the ball is not on anybody, in normalized frame units. A touch
# with nobody near it is the floor, not a player.
MAX_BALL_GAP = 0.06


def _nearest_sighting(ball_detections, frame_index, reach=8):
    for step in range(reach):
        for index in (frame_index - step, frame_index + step):
            if 0 <= index < len(ball_detections) and ball_detections[index] is not None:
                return ball_detections[index]
    return None


def _nearest_to(tracks, frame_index, point):
    best, best_gap = None, MAX_BALL_GAP
    for track in tracks:
        box = track.box(frame_index)
        if box is None:
            continue
        dx = max(box[0] - point[0], 0.0, point[0] - box[2])
        dy = max(box[1] - point[1], 0.0, point[1] - box[3])
        gap = math.hypot(dx, dy)
        if gap <= best_gap:
            best, best_gap = track, gap
    return best


def arrival_frame(target, contact_frame, fps):
    """First frame after contact where the target reaches up to play the ball.

    The target raises their hands to set or catch it, so the wrists crossing
    above the shoulders marks the arrival. That is the same geometry that tells
    a pass from an overhead action for the passer, read the other way round.

    ponytail: pose only, because it works with no ball track at all. Once the
    detector can see the ball, the arrival is just where the outgoing arc
    reaches the target and this becomes the fallback.
    """
    if target is None:
        return None

    horizon = contact_frame + int(round(MAX_FLIGHT_SECONDS * max(fps, 1)))
    for index, pose in zip(target.frame_indices, target.poses):
        if index <= contact_frame or index > horizon:
            continue
        height = wrists_above_shoulders(pose)
        if not math.isnan(height) and height >= TARGET_PLAY_HAND_HEIGHT:
            return index
    return None


def target_displacement(target, contact_frame, arrival_frame):
    """How far the target moved between contact and playing the ball.

    Measured from the contact frame on purpose. A setter moving into position
    before the pass is normal footwork, not a bad pass, so displacement from
    the start of the clip would punish the passer for it. Normalized by the
    target's own torso length, so it is scale free and camera independent.
    """
    if target is None:
        return None

    start = target.position(contact_frame)
    end = target.position(arrival_frame)
    if start is None or end is None:
        return None

    torso = _torso_length(target, contact_frame)
    if torso is None or torso < 0.001:
        return None
    return float(_distance(start, end) / torso)


def _torso_length(track, frame_index):
    for index, pose in zip(track.frame_indices, track.poses):
        if index != frame_index:
            continue
        shoulders = _midpoint(pose[LEFT_SIDE["shoulder"]], pose[RIGHT_SIDE["shoulder"]])
        hips = _midpoint(pose[LEFT_SIDE["hip"]], pose[RIGHT_SIDE["hip"]])
        return _distance(shoulders, hips)
    return None


def _hip_midpoint(pose):
    if pose is None or len(pose) < 33:
        return None
    return _midpoint(pose[LEFT_SIDE["hip"]], pose[RIGHT_SIDE["hip"]])


def _midpoint(a, b):
    return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])
