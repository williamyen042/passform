import math
from collections import Counter

import numpy as np

from core.angle_calculator import (
    axis_angle_difference,
    distance,
    joint_angle,
    midpoint,
    segment_angle_to_floor,
    segment_heading,
)


LEFT_SIDE = {
    "shoulder": 11,
    "elbow": 13,
    "wrist": 15,
    "hip": 23,
    "knee": 25,
    "ankle": 27,
}

RIGHT_SIDE = {
    "shoulder": 12,
    "elbow": 14,
    "wrist": 16,
    "hip": 24,
    "knee": 26,
    "ankle": 28,
}

FOOT_LANDMARKS = (27, 28, 29, 30, 31, 32)

PRE_CONTACT_SECONDS = 0.5
POST_CONTACT_SECONDS = 0.2
KINETIC_PRE_CONTACT_SECONDS = 0.15
# The ball is off the platform within a frame or two of contact, so a window
# reaching a quarter second past it measures the recovery, not the pass. On
# rep_0063 the shoulder drifted 38 degrees over that window, almost all of it
# after the ball had gone, and the kinetic score read 27 for a rep labelled 3.
KINETIC_POST_CONTACT_SECONDS = 0.12
SMOOTHING_RADIUS = 2
BALL_GAP_INTERPOLATION_FRAMES = 3
BALL_CONTACT_DISTANCE_GATE = 1.25
BALL_CONTACT_VELOCITY_WINDOW = 3
BALL_CONTACT_MIN_REVERSAL = 0.05
# Every ratio the scorer reports is measured in torso lengths. Shoulder width
# used to play that role and cannot: on the locked diagonal camera the passer
# turns to face the ball, their shoulders go edge-on, and the projected width
# collapses toward zero. Floored at 0.001 that reported a wrist gap of 0.09 as
# "30 shoulder widths", and put feet 13 shoulder widths apart in the dataset.
# Torso length runs along the body's long axis, so yaw barely shortens it.
MIN_BODY_SCALE = 0.02
# The bands below were written when ratios were divided by standing height or
# by shoulder width. Everything is in torso lengths now, so they are converted
# rather than re-invented - a band that was wrong before is still wrong, just
# no longer wrong about its units as well. Standing height is 2.45 torso
# lengths, measured over 891 frames of murphy footage. Shoulder width is taken
# at its anatomical 0.8 and not at the 0.18 those same frames show, because
# that 0.18 is the projection collapsing, which is why it stopped being the
# scale in the first place.
FROM_HEIGHT = 2.45
FROM_SHOULDERS = 0.8
# A passer cannot physically produce two contacts closer together than this, so
# it doubles as the non-max-suppression window for rep segmentation.
MIN_REP_SEPARATION_SECONDS = 0.8
# Secondary peaks count as reps only if they are nearly as strong as the best
# one. Keeps a single-rep clip returning exactly one rep.
REP_PEAK_MIN_FRACTION = 0.75
# ...and only if the signal genuinely falls away between them. Height alone is
# not enough: a broad plateau has plenty of points above 75% of its own maximum
# and none of them are separate reps.
REP_VALLEY_FRACTION = 0.7


def analyze_frames(frames_landmarks, fps=30, ball_detections=None, contacts=None):
    """Analyze one uploaded pass rep from an ordered landmark sequence.

    contacts overrides the detection above: when the caller already knows which
    touch is the pass - because the ball track told it, and it picked the
    passer from that same touch - letting this function choose again can score
    a different moment than the one the pose was cropped for.
    """
    frames = list(frames_landmarks)
    ball_detections = _aligned_ball_detections(ball_detections, len(frames))
    fps = max(float(fps or 30), 1.0)
    valid_indices = [
        index for index, landmarks in enumerate(frames)
        if _has_full_pose(landmarks)
    ]

    if not valid_indices:
        return {
            "overall_score": 0,
            "summary": ["No pose landmarks were detected."],
            "reps": [],
        }

    hip_y = _hip_y_series(frames)
    smoothed_hip_y = _smooth_series(hip_y, SMOOTHING_RADIUS)
    platform_score = _smooth_series(_platform_score_series(frames), SMOOTHING_RADIUS)
    rep_signal = _combined_rep_signal(smoothed_hip_y, platform_score, valid_indices)
    if contacts is None:
        contacts = _detect_contacts(
            rep_signal,
            valid_indices,
            frames,
            ball_detections,
            fps,
        )
    reps = []
    for contact in contacts:
        # Numbered after the drop, not before: a rep rejected for running off
        # the end of the clip used to leave a hole in the sequence, and these
        # indices become rep ids in the dataset.
        rep = _score_rep(len(reps) + 1, contact["frame_index"], frames, fps, contact)
        if rep is not None:
            reps.append(rep)

    if not reps:
        return {
            "overall_score": 0,
            "summary": ["Pose was detected, but no passing reps were found."],
            "reps": [],
        }

    overall_score = _round_score(np.mean([rep["scores"]["overall"] for rep in reps]))

    return {
        "overall_score": overall_score,
        "summary": _build_summary(reps),
        "contact_source": reps[-1]["contact_source"],
        "reps": reps,
    }


def _score_rep(rep_index, frame_center, frames, fps, contact=None):
    pre_contact_frames = max(1, int(round(PRE_CONTACT_SECONDS * fps)))
    post_contact_frames = max(1, int(round(POST_CONTACT_SECONDS * fps)))
    kinetic_pre_frames = max(1, int(round(KINETIC_PRE_CONTACT_SECONDS * fps)))
    kinetic_post_frames = max(1, int(round(KINETIC_POST_CONTACT_SECONDS * fps)))

    # The wider window is context and may legitimately be clipped, but the
    # kinetic window is the measurement itself. Without it there is nothing to
    # measure, and scoring anyway produces a confident number from almost no
    # data: murphy1 reported a rep at frame 0 this way, with no approach at all.
    if frame_center < kinetic_pre_frames:
        return None
    if frame_center > len(frames) - 1 - kinetic_post_frames:
        return None

    frame_start = max(0, frame_center - pre_contact_frames)
    frame_end = min(len(frames) - 1, frame_center + post_contact_frames)
    contact_start = max(0, frame_center - kinetic_pre_frames)
    contact_end = min(len(frames) - 1, frame_center + kinetic_post_frames)

    rep_window = [
        landmarks for landmarks in frames[frame_start:frame_end + 1]
        if _has_full_pose(landmarks)
    ]
    contact_window = [
        landmarks for landmarks in frames[contact_start:contact_end + 1]
        if _has_full_pose(landmarks)
    ]

    if not rep_window or not contact_window:
        return None

    scale = _window_scale(rep_window)
    if math.isnan(scale):
        return None

    side_name, side = _best_visible_side(contact_window)
    measurements = _measure_rep(rep_window, contact_window, side, scale)
    stability = _score_stability(measurements)
    integrity = _score_integrity(measurements)
    kinetic = _score_kinetic(measurements)
    overall = _score_overall_form(stability, integrity, kinetic)
    critiques = _build_critiques(measurements, stability, integrity, kinetic)

    return {
        "rep_index": rep_index,
        "frame_start": frame_start,
        "frame_center": frame_center,
        "frame_end": frame_end,
        "contact_start": contact_start,
        "contact_end": contact_end,
        "contact_source": _contact_value(contact, "contact_source", "pose_proxy"),
        "ball_contact_distance": _contact_value(contact, "ball_contact_distance"),
        "ball_confidence": _contact_value(contact, "ball_confidence"),
        "side_used": side_name,
        "scores": {
            "stability": stability,
            "integrity": integrity,
            "kinetic": kinetic,
            "overall": overall,
        },
        # Pose-only hint for the 0-3 passing scale. This should be combined
        # with ball/target outcome later, once detection exists.
        "form_pass_quality_hint": _form_score_to_pass_quality(overall),
        "measurements": measurements,
        "critiques": critiques,
    }


def _measure_rep(rep_window, contact_window, side, scale):
    shoulder_angles = []
    shoulder_y_values = []
    hip_y_values = []
    rep_hip_y_values = []
    wrist_y_values = []
    elbow_angles = []
    forearm_headings = []
    nose_y_values = []
    frame_measurements = []

    for landmarks in rep_window:
        left_shoulder = landmarks[LEFT_SIDE["shoulder"]]
        right_shoulder = landmarks[RIGHT_SIDE["shoulder"]]
        left_hip = landmarks[LEFT_SIDE["hip"]]
        right_hip = landmarks[RIGHT_SIDE["hip"]]
        left_wrist = landmarks[LEFT_SIDE["wrist"]]
        right_wrist = landmarks[RIGHT_SIDE["wrist"]]
        left_ankle = landmarks[LEFT_SIDE["ankle"]]
        right_ankle = landmarks[RIGHT_SIDE["ankle"]]

        shoulder_mid = midpoint(left_shoulder, right_shoulder)
        hip_mid = midpoint(left_hip, right_hip)
        ankle_y = max(left_ankle.y, right_ankle.y)

        # Stability metric: knee bend depth. Good passing form uses an athletic
        # knee bend, not locked legs and not an overly deep squat.
        knee_angle = joint_angle(
            landmarks[side["hip"]],
            landmarks[side["knee"]],
            landmarks[side["ankle"]],
        )

        # Integrity metric: elbow lock. A firm platform should keep the elbow
        # angle close to straight, roughly 170-180 degrees.
        elbow_angle = joint_angle(
            landmarks[side["wrist"]],
            landmarks[side["elbow"]],
            landmarks[side["shoulder"]],
        )

        # Integrity metric: the arms should be held away from the torso,
        # close to a right angle, instead of tucked into the stomach.
        arm_torso_angle = joint_angle(
            landmarks[side["hip"]],
            landmarks[side["shoulder"]],
            landmarks[side["wrist"]],
        )

        # Stability metric: torso posture. This estimates forward lean by
        # comparing each hip-to-shoulder segment against the floor line.
        torso_angle = segment_angle_to_floor(left_hip, left_shoulder)
        torso_angle = np.nanmean([
            torso_angle,
            segment_angle_to_floor(right_hip, right_shoulder),
        ])

        # Integrity metric: forearm parallelism. Parallel forearms create a
        # cleaner, more predictable passing platform.
        left_forearm = segment_heading(
            landmarks[LEFT_SIDE["elbow"]],
            landmarks[LEFT_SIDE["wrist"]],
        )
        right_forearm = segment_heading(
            landmarks[RIGHT_SIDE["elbow"]],
            landmarks[RIGHT_SIDE["wrist"]],
        )
        forearm_parallel_delta = axis_angle_difference(left_forearm, right_forearm)

        frame_measurements.append({
            "knee_angle": knee_angle,
            "elbow_angle": elbow_angle,
            "arm_torso_angle": arm_torso_angle,
            "torso_angle": torso_angle,
            "forearm_parallel_delta": forearm_parallel_delta,
            # Integrity metric: hands should be together before contact so
            # the ball sees one flat, predictable platform.
            "wrist_gap_ratio": distance(left_wrist, right_wrist) / scale,
            # Stability metric: rough center-of-gravity depth. Higher values
            # mean the hips are lower relative to the feet and shoulders.
            "cog_ratio": (ankle_y - hip_mid[1]) / scale,
            # Stability metric: projected body mass over the support base.
            # Lower values mean the estimated mass sits closer to the middle
            # of the feet, which is better for receiving instead of reaching.
            "balance_offset": _projected_balance_offset(landmarks, scale),
            # Extra critique metric: compares foot width to shoulder width so
            # the scorer can flag a base that is too narrow or too wide.
            "stance_width_ratio": distance(left_ankle, right_ankle) / scale,
            # Extra critique metric: large horizontal shoulder/hip separation
            # can indicate torso twist or loss of body alignment.
            "shoulder_hip_offset": abs(shoulder_mid[0] - hip_mid[0]) / scale,
        })
        rep_hip_y_values.append(hip_mid[1])

    for landmarks in contact_window:
        left_shoulder = landmarks[LEFT_SIDE["shoulder"]]
        right_shoulder = landmarks[RIGHT_SIDE["shoulder"]]
        left_hip = landmarks[LEFT_SIDE["hip"]]
        right_hip = landmarks[RIGHT_SIDE["hip"]]
        left_wrist = landmarks[LEFT_SIDE["wrist"]]
        right_wrist = landmarks[RIGHT_SIDE["wrist"]]

        # Kinetic metric: shoulder angle change is now only one part of the
        # score. Movement is allowed when it stays connected to hips/legs.
        shoulder_angles.append(joint_angle(
            landmarks[side["hip"]],
            landmarks[side["shoulder"]],
            landmarks[side["elbow"]],
        ))
        shoulder_y_values.append(midpoint(left_shoulder, right_shoulder)[1])
        hip_y_values.append(midpoint(left_hip, right_hip)[1])
        wrist_y_values.append(midpoint(left_wrist, right_wrist)[1])
        elbow_angles.append(joint_angle(
            landmarks[side["wrist"]],
            landmarks[side["elbow"]],
            landmarks[side["shoulder"]],
        ))
        nose_y_values.append(landmarks[0].y)
        forearm_headings.append(segment_heading(
            landmarks[side["elbow"]],
            landmarks[side["wrist"]],
        ))

    measurements = {
        key: _safe_mean([values[key] for values in frame_measurements])
        for key in frame_measurements[0]
    }
    # Kinetic metric: total shoulder angle movement across the contact window.
    measurements["shoulder_delta"] = _safe_range(shoulder_angles)
    # Kinetic metric: shoulder and hip vertical movement should match. If the
    # shoulders rise much more than the hips, the arms are likely swinging.
    measurements["shoulder_hip_sync_error"] = _relative_motion_error(
        shoulder_y_values,
        hip_y_values,
        scale,
    )
    # Kinetic metric: platform/wrists should travel with the shoulders instead
    # of whipping independently through contact.
    measurements["platform_shoulder_sync_error"] = _relative_motion_error(
        wrist_y_values,
        shoulder_y_values,
        scale,
    )
    # Integrity-through-contact metric: elbows should stay locked, not flex and
    # extend rapidly as the ball arrives.
    measurements["elbow_delta"] = _safe_range(elbow_angles)
    # Platform stability metric: forearm direction should not rotate sharply
    # during the contact window.
    measurements["forearm_angle_delta"] = _angle_range(forearm_headings)
    # Stability metric: too much vertical rise means the player popped up
    # through the pass instead of staying controlled after contact.
    #
    # Measured over the contact window, not the rep window. The rep window
    # opens half a second before contact, which is a passer moving to the
    # ball, so this was scoring normal footwork as popping up: on rep_0063 it
    # read 0.96 torso lengths against a limit of 0.69 and took the whole
    # stability term to zero for a rep labelled 3.
    measurements["body_rise"] = _safe_range(hip_y_values) / scale
    # Extra critique metric: head bob through contact, same window and same
    # reason as body_rise above.
    measurements["head_y_delta"] = _safe_range(nose_y_values) / scale

    return {
        key: float(round(float(value), 3)) if not math.isnan(value) else None
        for key, value in measurements.items()
    }


def _score_stability(measurements):
    # Stability combines lower-body loading, body height, and torso posture.
    knee_score = _score_target_range(measurements["knee_angle"], 120, 155, 90, 180)
    cog_score = _score_target_range(
        measurements["cog_ratio"],
        0.32 * FROM_HEIGHT, 0.52 * FROM_HEIGHT,
        0.15 * FROM_HEIGHT, 0.75 * FROM_HEIGHT)
    balance_score = _score_max_allowed(
        measurements["balance_offset"], 0.22 * FROM_SHOULDERS, 0.55 * FROM_SHOULDERS)
    torso_score = _score_target_range(measurements["torso_angle"], 50, 80, 25, 90)
    rise_score = _score_max_allowed(
        measurements["body_rise"], 0.10 * FROM_HEIGHT, 0.28 * FROM_HEIGHT)
    return _round_score(
        (knee_score * 0.25)
        + (cog_score * 0.20)
        + (balance_score * 0.25)
        + (torso_score * 0.20)
        + (rise_score * 0.10)
    )


def _score_integrity(measurements):
    # Integrity focuses on whether the platform is straight and even.
    elbow_score = _score_target_range(measurements["elbow_angle"], 170, 180, 135, 180)
    arm_torso_score = _score_target_range(
        measurements["arm_torso_angle"],
        75,
        115,
        35,
        150,
    )
    wrist_score = _score_max_allowed(
        measurements["wrist_gap_ratio"], 0.65 * FROM_SHOULDERS, 1.4 * FROM_SHOULDERS)
    parallel_score = _score_max_allowed(measurements["forearm_parallel_delta"], 10, 45)
    return _round_score(
        (elbow_score * 0.35)
        + (parallel_score * 0.25)
        + (wrist_score * 0.20)
        + (arm_torso_score * 0.20)
    )


def _score_kinetic(measurements):
    # Kinetic rewards connected movement: shoulders/platform can rise, but
    # they should move with hips/legs instead of breaking away as arm swing.
    shoulder_sync = _score_max_allowed(
        measurements["shoulder_hip_sync_error"],
        0.05 * FROM_HEIGHT,
        0.22 * FROM_HEIGHT,
    )
    platform_sync = _score_max_allowed(
        measurements["platform_shoulder_sync_error"],
        0.04 * FROM_HEIGHT,
        0.18 * FROM_HEIGHT,
    )
    shoulder_stability = _score_max_allowed(measurements["shoulder_delta"], 18, 55)
    elbow_stability = _score_max_allowed(measurements["elbow_delta"], 8, 30)
    forearm_stability = _score_max_allowed(measurements["forearm_angle_delta"], 10, 35)
    return _round_score(
        (shoulder_sync * 0.35)
        + (platform_sync * 0.25)
        + (elbow_stability * 0.20)
        + (forearm_stability * 0.15)
        + (shoulder_stability * 0.05)
    )


def _score_overall_form(stability, integrity, kinetic):
    # Form-only score aligned with the future 0-3 pass scale: strong platform
    # and ready-position form should make a "3" outcome plausible, while poor
    # platform integrity should keep lucky passes from scoring too high.
    return _round_score(
        (stability * 0.35)
        + (integrity * 0.40)
        + (kinetic * 0.25)
    )


def _form_score_to_pass_quality(score):
    if score >= 85:
        return 3
    if score >= 70:
        return 2
    if score >= 45:
        return 1
    return 0


def _build_critiques(measurements, stability, integrity, kinetic):
    """Faults first, worst dimension first, with any praise last.

    Callers show critiques[0] and nothing else, and these used to come out in
    source order with the elbow compliment written first. A rep with good
    elbows therefore displayed praise while five real faults sat unread behind
    it.
    """
    ranked = sorted(
        (
            (stability, _stability_critiques(measurements)),
            (integrity, _integrity_critiques(measurements)),
            (kinetic, _kinetic_critiques(measurements, kinetic)),
        ),
        key=lambda item: item[0],
    )
    critiques = [note for _, notes in ranked for note in notes]
    return critiques + _praise(measurements, stability, integrity, kinetic)


def _integrity_critiques(measurements):
    critiques = []
    elbow_angle = _measurement(measurements, "elbow_angle")
    arm_torso_angle = _measurement(measurements, "arm_torso_angle")
    forearm_parallel_delta = _measurement(measurements, "forearm_parallel_delta")
    wrist_gap_ratio = _measurement(measurements, "wrist_gap_ratio")

    if elbow_angle < 170 and not math.isnan(elbow_angle):
        critiques.append("Lock the elbows more so the platform stays firm.")

    if forearm_parallel_delta > 25:
        critiques.append("Bring the forearms closer to parallel before contact.")
    if wrist_gap_ratio > 1.4 * FROM_SHOULDERS:
        critiques.append("Bring the hands together earlier to create one platform.")
    elif wrist_gap_ratio > 0.8 * FROM_SHOULDERS:
        critiques.append("Close the wrist gap so the platform is flatter.")
    if arm_torso_angle < 65:
        critiques.append("Hold the platform away from the stomach before contact.")
    elif arm_torso_angle > 130:
        critiques.append("Set the platform closer to a right angle with the torso.")

    return critiques


def _stability_critiques(measurements):
    critiques = []
    knee_angle = _measurement(measurements, "knee_angle")
    torso_angle = _measurement(measurements, "torso_angle")
    stance_width_ratio = _measurement(measurements, "stance_width_ratio")
    balance_offset = _measurement(measurements, "balance_offset")
    head_y_delta = _measurement(measurements, "head_y_delta")
    shoulder_hip_offset = _measurement(measurements, "shoulder_hip_offset")
    body_rise = _measurement(measurements, "body_rise")

    if knee_angle > 160:
        critiques.append("Bend the knees more before contact to load the legs.")
    elif knee_angle < 110:
        critiques.append("Avoid dropping too deep; stay balanced and ready to extend.")

    if torso_angle > 82:
        critiques.append("Lean slightly forward instead of staying upright.")
    elif torso_angle < 40:
        critiques.append("Keep the chest from collapsing too far over the platform.")
    if body_rise > 0.28 * FROM_HEIGHT:
        critiques.append("Avoid popping up after contact; stay low and controlled.")

    if balance_offset > 0.55 * FROM_SHOULDERS:
        critiques.append("Keep your body mass inside your feet instead of reaching for the ball.")
    elif balance_offset > 0.35 * FROM_SHOULDERS:
        critiques.append("Center your weight more evenly over your base before contact.")

    if stance_width_ratio < 1.0 * FROM_SHOULDERS:
        critiques.append("Widen the stance for a more stable passing base.")
    elif stance_width_ratio > 2.8 * FROM_SHOULDERS:
        critiques.append("Narrow the stance slightly so you can move after the pass.")

    if head_y_delta > 0.08 * FROM_HEIGHT:
        critiques.append("Keep the head quieter through contact to improve control.")

    if shoulder_hip_offset > 1.2 * FROM_SHOULDERS:
        critiques.append("Keep shoulders and hips more connected through the pass.")

    return critiques


def _kinetic_critiques(measurements, kinetic):
    critiques = []
    shoulder_hip_sync_error = _measurement(measurements, "shoulder_hip_sync_error")
    platform_shoulder_sync_error = _measurement(
        measurements,
        "platform_shoulder_sync_error",
    )
    elbow_delta = _measurement(measurements, "elbow_delta")
    forearm_angle_delta = _measurement(measurements, "forearm_angle_delta")

    if kinetic < 75:
        critiques.append("Keep the platform connected to the legs through contact.")
    if shoulder_hip_sync_error > 0.16 * FROM_HEIGHT:
        critiques.append("Shoulders rose separately from the hips; drive more from the legs.")
    if platform_shoulder_sync_error > 0.18 * FROM_HEIGHT:
        critiques.append("Platform moved independently of the shoulders during contact.")
    if elbow_delta > 30:
        critiques.append("Elbow angle changed too much at contact; keep the platform locked.")
    if forearm_angle_delta > 35:
        critiques.append("Forearms rotated through contact; hold the platform angle steady.")

    return critiques


def _praise(measurements, stability, integrity, kinetic):
    """Only ever shown once the faults above it have run out."""
    notes = []
    elbow_angle = _measurement(measurements, "elbow_angle")
    if elbow_angle >= 170:
        notes.append("Strong elbow extension through the platform.")
    if stability >= 85 and integrity >= 85 and kinetic >= 85:
        notes.append("Balanced rep with stable legs, locked elbows, and quiet shoulders.")
    return notes


def _measurement(measurements, key):
    value = measurements.get(key)
    if value is None:
        return float("nan")
    return value


def _build_summary(reps):
    all_critiques = [
        critique for rep in reps
        for critique in rep["critiques"]
    ]
    counts = Counter(all_critiques)
    return [critique for critique, _ in counts.most_common(3)]


def _detect_contacts(rep_signal, valid_indices, frames, ball_detections, fps):
    """Return one contact descriptor per detected rep, ordered by frame."""
    separation = max(1, int(round(MIN_REP_SEPARATION_SECONDS * fps)))
    ball_contacts = _detect_ball_contacts(
        frames,
        ball_detections,
        valid_indices,
        separation,
    )
    if ball_contacts:
        return ball_contacts

    return [
        {
            "frame_index": frame_index,
            "contact_source": "pose_proxy",
            "ball_contact_distance": None,
            "ball_confidence": None,
        }
        for frame_index in _detect_pose_contact_centers(
            rep_signal,
            valid_indices,
            separation,
        )
    ]


def _detect_pose_contact_centers(rep_signal, valid_indices, separation):
    """Pick pose-only contact estimates: greedy peaks, one rep apart.

    ponytail: greedy peak picking, not a real segmenter. It cannot split two
    passes that blend into each other, and REP_PEAK_MIN_FRACTION is a guess.
    The hand-labeled contact frames from the dataset build are what tell us
    whether this needs replacing.
    """
    valid_values = rep_signal[valid_indices]
    best = float(np.nanmax(valid_values))
    if math.isclose(best, float(np.nanmin(valid_values))):
        # Nothing to go on. The middle of the clip is the least bad guess, and
        # unlike argmax (which lands on frame 0) it has a window either side.
        return [int(valid_indices[len(valid_indices) // 2])]

    floor = best * REP_PEAK_MIN_FRACTION
    remaining = {
        int(frame_index): float(value)
        for frame_index, value in zip(valid_indices, valid_values)
        if not math.isnan(value)
    }
    centers = []
    while remaining:
        frame_index = max(remaining, key=remaining.get)
        if remaining[frame_index] < floor:
            break
        if _separated_by_a_valley(rep_signal, centers, frame_index):
            centers.append(frame_index)
        remaining = {
            other: value
            for other, value in remaining.items()
            if abs(other - frame_index) >= separation
        }

    return sorted(centers)


def _separated_by_a_valley(rep_signal, accepted, candidate):
    """True when the signal dips away between this peak and every earlier one."""
    for peak in accepted:
        low = min(candidate, peak)
        high = max(candidate, peak)
        valley = np.nanmin(rep_signal[low:high + 1])
        if valley > REP_VALLEY_FRACTION * min(rep_signal[peak], rep_signal[candidate]):
            return False
    return True


def _detect_ball_contacts(frames, ball_detections, valid_indices, separation):
    if not ball_detections or not any(ball_detections):
        return []

    ball_track = _interpolated_ball_track(ball_detections)
    metrics = {}
    for index in valid_indices:
        landmarks = frames[index]
        ball = ball_track[index] if index < len(ball_track) else None
        if ball is None or not _has_full_pose(landmarks):
            continue

        scale = body_scale(landmarks)
        if math.isnan(scale):
            continue
        platform_point = _platform_point(landmarks)
        ball_center = np.array(ball["center"], dtype=float)
        distance_ratio = float(np.linalg.norm(ball_center - platform_point) / scale)
        metrics[index] = {
            "distance_ratio": distance_ratio,
            "confidence": ball["confidence"],
            "interpolated": ball["interpolated"],
        }

    candidates = []
    for index, current in metrics.items():
        if current["distance_ratio"] > BALL_CONTACT_DISTANCE_GATE:
            continue

        previous = _window_endpoint_metric(metrics, index, -BALL_CONTACT_VELOCITY_WINDOW)
        following = _window_endpoint_metric(metrics, index, BALL_CONTACT_VELOCITY_WINDOW)
        if previous is None or following is None:
            continue

        approach = previous["distance_ratio"] - current["distance_ratio"]
        depart = following["distance_ratio"] - current["distance_ratio"]
        if approach < BALL_CONTACT_MIN_REVERSAL or depart < BALL_CONTACT_MIN_REVERSAL:
            continue

        reversal_strength = approach + depart
        score = (
            current["distance_ratio"]
            - (current["confidence"] * 0.15)
            - (min(reversal_strength, 1.0) * 0.25)
            + (0.05 if current["interpolated"] else 0.0)
        )
        candidates.append({
            "frame_index": int(index),
            "contact_source": "ball",
            "ball_contact_distance": float(round(current["distance_ratio"], 3)),
            "ball_confidence": float(round(current["confidence"], 3)),
            "_score": score,
        })

    # Lower _score is a better contact, so keep the best candidate in each
    # separation window and drop the rest.
    chosen = []
    for candidate in sorted(candidates, key=lambda item: item["_score"]):
        if all(
            abs(candidate["frame_index"] - other["frame_index"]) >= separation
            for other in chosen
        ):
            chosen.append(candidate)

    for candidate in chosen:
        candidate.pop("_score", None)
    return sorted(chosen, key=lambda item: item["frame_index"])


def _aligned_ball_detections(ball_detections, frame_count):
    if ball_detections is None:
        return [None] * frame_count

    aligned = list(ball_detections)[:frame_count]
    if len(aligned) < frame_count:
        aligned.extend([None] * (frame_count - len(aligned)))
    return aligned


def _interpolated_ball_track(ball_detections):
    track = [_ball_track_entry(detection) for detection in ball_detections]
    valid_indices = [index for index, entry in enumerate(track) if entry is not None]

    for left_index, right_index in zip(valid_indices, valid_indices[1:]):
        gap = right_index - left_index - 1
        if gap <= 0 or gap > BALL_GAP_INTERPOLATION_FRAMES:
            continue

        left = track[left_index]
        right = track[right_index]
        for offset in range(1, gap + 1):
            ratio = offset / (gap + 1)
            center = (
                (left["center"][0] * (1.0 - ratio)) + (right["center"][0] * ratio),
                (left["center"][1] * (1.0 - ratio)) + (right["center"][1] * ratio),
            )
            track[left_index + offset] = {
                "center": center,
                "confidence": min(left["confidence"], right["confidence"]) * 0.8,
                "interpolated": True,
            }

    return track


def _ball_track_entry(detection):
    if detection is None:
        return None

    center = _ball_detection_value(detection, "center")
    if center is None or len(center) != 2:
        return None

    confidence = _ball_detection_value(detection, "confidence", 0.0)
    return {
        "center": (float(center[0]), float(center[1])),
        "confidence": float(confidence or 0.0),
        "interpolated": False,
    }


def _ball_detection_value(detection, key, default=None):
    if isinstance(detection, dict):
        return detection.get(key, default)
    return getattr(detection, key, default)


def _contact_value(contact, key, default=None):
    if contact is None:
        return default
    return contact.get(key, default)


def contact_geometry(landmarks, ball_center):
    """Where the ball met the passer, relative to the passer.

    A pass needs room. When the ball arrives into the chest rather than out in
    front, there is no angle to play it from however good the platform is, and
    the result is weak regardless of form - which is a thing the pose features
    cannot see, because the body can look identical either way.

    reach   distance from the torso centre, in torso lengths. Small is jammed.
    height  where it met them, in torso lengths above the hips. Negative is
            below the waist, around 1 is chest high.
    ahead   how far in front of the torso, signed toward the platform, so a
            ball met behind the body reads negative.
    """
    if not _has_full_pose(landmarks) or ball_center is None:
        return {"contact_reach": None, "contact_height": None, "contact_ahead": None}
    scale = body_scale(landmarks)
    if math.isnan(scale):
        return {"contact_reach": None, "contact_height": None, "contact_ahead": None}

    shoulder_mid = midpoint(
        landmarks[LEFT_SIDE["shoulder"]], landmarks[RIGHT_SIDE["shoulder"]])
    hip_mid = midpoint(landmarks[LEFT_SIDE["hip"]], landmarks[RIGHT_SIDE["hip"]])
    torso_centre = (shoulder_mid + hip_mid) / 2.0
    ball = np.array(ball_center, dtype=float)

    facing = _platform_point(landmarks) - torso_centre
    length = float(np.linalg.norm(facing))
    ahead = (float(np.dot(ball - torso_centre, facing / length)) / scale
             if length > 1e-6 else None)
    return {
        "contact_reach": round(float(np.linalg.norm(ball - torso_centre)) / scale, 3),
        "contact_height": round(float(hip_mid[1] - ball[1]) / scale, 3),
        "contact_ahead": round(ahead, 3) if ahead is not None else None,
    }


def body_scale(landmarks):
    """Torso length: the one length that holds up when the passer turns.

    NaN rather than a floored constant when the pose is too small or too broken
    to measure. A floor turns "unmeasurable" into a confident wrong number, and
    everything downstream then treats that number as data.
    """
    if not _has_full_pose(landmarks):
        return float("nan")
    shoulder_mid = midpoint(
        landmarks[LEFT_SIDE["shoulder"]], landmarks[RIGHT_SIDE["shoulder"]],
    )
    hip_mid = midpoint(landmarks[LEFT_SIDE["hip"]], landmarks[RIGHT_SIDE["hip"]])
    scale = float(np.linalg.norm(shoulder_mid - hip_mid))
    return scale if scale >= MIN_BODY_SCALE else float("nan")


def _window_scale(window):
    """One scale for the whole rep, so a single bad frame cannot blow up a feature."""
    values = [
        value for value in (body_scale(landmarks) for landmarks in window)
        if not math.isnan(value)
    ]
    return float(np.median(values)) if values else float("nan")


def _platform_point(landmarks):
    return midpoint(
        landmarks[LEFT_SIDE["wrist"]],
        landmarks[RIGHT_SIDE["wrist"]],
    )


def _window_endpoint_metric(metrics, index, offset):
    step = 1 if offset > 0 else -1
    for probe in range(index + offset, index, -step):
        metric = metrics.get(probe)
        if metric is not None:
            return metric
    return None


def _combined_rep_signal(hip_y, platform_score, valid_indices):
    # Rep detection proxy: a rep is most likely when the athlete is low and the
    # platform is formed. Future ball tracking can replace this contact proxy.
    hip_component = _normalize_series(hip_y, valid_indices)
    platform_component = _normalize_series(platform_score, valid_indices)
    return (0.6 * hip_component) + (0.4 * platform_component)


def wrists_above_shoulders(landmarks):
    """How far the hands are raised above the shoulders, in torso lengths.

    Zero whenever the hands are at or below shoulder height, so a pass reads as
    zero and an overhead play reads positive. Normalized by torso length to
    stay scale free.
    """
    if not _has_full_pose(landmarks):
        return float("nan")

    torso_length = max(
        distance(landmarks[LEFT_SIDE["shoulder"]], landmarks[LEFT_SIDE["hip"]]),
        0.001,
    )
    shoulder_y = (
        landmarks[LEFT_SIDE["shoulder"]].y + landmarks[RIGHT_SIDE["shoulder"]].y
    ) / 2.0
    wrist_y = (
        landmarks[LEFT_SIDE["wrist"]].y + landmarks[RIGHT_SIDE["wrist"]].y
    ) / 2.0
    return max((shoulder_y - wrist_y) / torso_length, 0.0)


def platform_score(landmarks):
    """How much a pose looks like a formed passing platform, 0-100.

    Wrists together and forearms parallel are not enough on their own: a person
    standing idle with their arms at their sides scores about 85 on those two
    alone. The arm-to-torso term is what separates a passer from anyone else in
    the frame, so this doubles as the passer/target discriminator.
    """
    if not _has_full_pose(landmarks):
        return float("nan")

    shoulder_width = max(
        distance(landmarks[LEFT_SIDE["shoulder"]], landmarks[RIGHT_SIDE["shoulder"]]),
        0.001,
    )
    wrist_gap = (
        distance(landmarks[LEFT_SIDE["wrist"]], landmarks[RIGHT_SIDE["wrist"]])
        / shoulder_width
    )
    left_forearm = segment_heading(
        landmarks[LEFT_SIDE["elbow"]],
        landmarks[LEFT_SIDE["wrist"]],
    )
    right_forearm = segment_heading(
        landmarks[RIGHT_SIDE["elbow"]],
        landmarks[RIGHT_SIDE["wrist"]],
    )
    parallel_delta = axis_angle_difference(left_forearm, right_forearm)
    # A pass is played below the shoulders. Sets, serves and spikes are not,
    # and without this term an overhead jump serve scores as a passing rep:
    # the hips are low on landing and the arms briefly look like a platform.
    hand_height = wrists_above_shoulders(landmarks)

    arm_torso_angle = float(np.nanmean([
        joint_angle(
            landmarks[LEFT_SIDE["hip"]],
            landmarks[LEFT_SIDE["shoulder"]],
            landmarks[LEFT_SIDE["wrist"]],
        ),
        joint_angle(
            landmarks[RIGHT_SIDE["hip"]],
            landmarks[RIGHT_SIDE["shoulder"]],
            landmarks[RIGHT_SIDE["wrist"]],
        ),
    ]))

    return float(np.mean([
        _score_max_allowed(wrist_gap, 0.75, 2.0),
        _score_max_allowed(parallel_delta, 15, 60),
        _score_target_range(arm_torso_angle, 55, 120, 15, 165),
        _score_max_allowed(hand_height, 0.0, 0.5),
    ]))


def _platform_score_series(frames):
    values = np.full(len(frames), np.nan, dtype=float)
    for index, landmarks in enumerate(frames):
        values[index] = platform_score(landmarks)
    return values


def _normalize_series(values, valid_indices):
    normalized = np.full(len(values), np.nan, dtype=float)
    valid_values = values[valid_indices]
    minimum = np.nanmin(valid_values)
    maximum = np.nanmax(valid_values)

    if math.isclose(maximum, minimum):
        normalized[valid_indices] = 0.5
        return normalized

    normalized[valid_indices] = (valid_values - minimum) / (maximum - minimum)
    return _fill_missing(normalized)


def _hip_y_series(frames):
    values = np.full(len(frames), np.nan, dtype=float)
    for index, landmarks in enumerate(frames):
        if _has_full_pose(landmarks):
            values[index] = (
                landmarks[LEFT_SIDE["hip"]].y + landmarks[RIGHT_SIDE["hip"]].y
            ) / 2.0
    return values


def _smooth_series(values, radius):
    filled = _fill_missing(values)
    smoothed = np.full(len(filled), np.nan, dtype=float)

    for index in range(len(filled)):
        left = max(0, index - radius)
        right = min(len(filled), index + radius + 1)
        smoothed[index] = np.nanmean(filled[left:right])

    return smoothed


def _fill_missing(values):
    values = np.asarray(values, dtype=float)
    valid = np.flatnonzero(~np.isnan(values))
    if len(valid) == 0:
        return values
    if len(valid) == len(values):
        return values

    return np.interp(np.arange(len(values)), valid, values[valid])


def _best_visible_side(window):
    left_visibility = _side_visibility(window, LEFT_SIDE)
    right_visibility = _side_visibility(window, RIGHT_SIDE)
    if left_visibility >= right_visibility:
        return "left", LEFT_SIDE
    return "right", RIGHT_SIDE


def _projected_balance_offset(landmarks, scale):
    support_x_values = [
        landmarks[index].x
        for index in FOOT_LANDMARKS
        if getattr(landmarks[index], "visibility", 1.0) >= 0.35
    ]
    if len(support_x_values) < 2:
        support_x_values = [
            landmarks[LEFT_SIDE["ankle"]].x,
            landmarks[RIGHT_SIDE["ankle"]].x,
        ]

    support_left = min(support_x_values)
    support_right = max(support_x_values)
    support_width = support_right - support_left

    shoulder_mid = midpoint(
        landmarks[LEFT_SIDE["shoulder"]],
        landmarks[RIGHT_SIDE["shoulder"]],
    )
    hip_mid = midpoint(
        landmarks[LEFT_SIDE["hip"]],
        landmarks[RIGHT_SIDE["hip"]],
    )
    knee_mid = midpoint(
        landmarks[LEFT_SIDE["knee"]],
        landmarks[RIGHT_SIDE["knee"]],
    )
    ankle_mid = midpoint(
        landmarks[LEFT_SIDE["ankle"]],
        landmarks[RIGHT_SIDE["ankle"]],
    )
    projected_mass_x = (
        (landmarks[0].x * 0.07)
        + (shoulder_mid[0] * 0.23)
        + (hip_mid[0] * 0.42)
        + (knee_mid[0] * 0.20)
        + (ankle_mid[0] * 0.08)
    )

    # Measured in torso lengths, not in widths of the base itself: feet seen
    # end-on have almost no projected separation, and dividing by that turned
    # a normal stance into an offset of 9.
    support_center = (support_left + support_right) / 2.0
    return abs(projected_mass_x - support_center) / scale


def _side_visibility(window, side):
    visibility_values = []
    for landmarks in window:
        for index in side.values():
            visibility_values.append(getattr(landmarks[index], "visibility", 1.0))
    return _safe_mean(visibility_values)


def _has_full_pose(landmarks):
    return landmarks is not None and len(landmarks) >= 33


def _score_target_range(value, low, high, floor, ceiling):
    if value is None or math.isnan(value):
        return 0
    if low <= value <= high:
        return 100
    if value < low:
        return _score_between(value, floor, low)
    return _score_between(value, ceiling, high)


def _score_max_allowed(value, ideal_max, hard_max):
    if value is None or math.isnan(value):
        return 0
    if value <= ideal_max:
        return 100
    if value >= hard_max:
        return 0
    return 100 * (hard_max - value) / (hard_max - ideal_max)


def _score_between(value, zero_at, full_at):
    if zero_at == full_at:
        return 0
    score = 100 * (value - zero_at) / (full_at - zero_at)
    return float(np.clip(score, 0, 100))


def _safe_mean(values):
    clean_values = [
        value for value in values
        if value is not None and not math.isnan(value)
    ]
    if not clean_values:
        return float("nan")
    return float(np.mean(clean_values))


def _median_filter(values, radius=2):
    """Impulse noise is what pose jitter is, and this is what removes it.

    A 5 percent trim on a 17 frame window clips less than one sample, so the
    dropped-joint frame still set the range through its neighbours. A median
    over five frames deletes it outright and leaves real movement untouched.
    """
    if len(values) < 2 * radius + 1:
        return list(values)
    filtered = []
    for index in range(len(values)):
        window = values[max(0, index - radius):index + radius + 1]
        filtered.append(float(np.median(window)))
    return filtered


def _safe_range(values, trim=5.0):
    """How much a series moved, ignoring single-frame spikes.

    MediaPipe drops a joint for one frame and puts it straight back: on
    rep_0063 the elbow read 180, 118, 158 across three consecutive frames,
    which no arm does in 33ms. max - min makes that one frame the whole
    measurement, so the 5th to 95th percentile is used instead. Still the
    spread, minus the physically impossible.
    """
    clean_values = [
        value for value in values
        if value is not None and not math.isnan(value)
    ]
    if not clean_values:
        return float("nan")
    if len(clean_values) < 5:
        return float(max(clean_values) - min(clean_values))
    clean_values = _median_filter(clean_values)
    return float(np.percentile(clean_values, 100 - trim)
                 - np.percentile(clean_values, trim))


def _relative_motion_error(primary_values, reference_values, scale):
    primary_delta = _motion_delta(primary_values)
    reference_delta = _motion_delta(reference_values)
    if math.isnan(primary_delta) or math.isnan(reference_delta):
        return float("nan")
    return abs(primary_delta - reference_delta) / scale


def _motion_delta(values):
    clean_values = [
        value for value in values
        if value is not None and not math.isnan(value)
    ]
    if len(clean_values) < 2:
        return float("nan")
    return float(clean_values[-1] - clean_values[0])


def _angle_range(values, trim=5.0):
    """_safe_range for headings, which wrap at 360.

    Anchored on the median rather than the first value: anchoring on the first
    made a jittered opening frame define the whole window, which is the same
    way one bad frame used to define an elbow's range.
    """
    clean_values = [
        value for value in values
        if value is not None and not math.isnan(value)
    ]
    if not clean_values:
        return float("nan")

    anchor = float(np.median(clean_values))
    relative_values = [
        (value - anchor + 180.0) % 360.0 - 180.0
        for value in clean_values
    ]
    if len(relative_values) < 5:
        return float(max(relative_values) - min(relative_values))
    relative_values = _median_filter(relative_values)
    return float(np.percentile(relative_values, 100 - trim)
                 - np.percentile(relative_values, trim))


def _round_score(value):
    if value is None or math.isnan(value):
        return 0
    return int(round(float(np.clip(value, 0, 100))))
