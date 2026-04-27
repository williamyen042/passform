import math
from collections import Counter

import numpy as np

from core.angle_calculator import (
    angle_difference,
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

PRE_CONTACT_SECONDS = 0.5
POST_CONTACT_SECONDS = 0.2
KINETIC_PRE_CONTACT_SECONDS = 0.15
KINETIC_POST_CONTACT_SECONDS = 0.25
SMOOTHING_RADIUS = 2


def analyze_frames(frames_landmarks, fps=30):
    """Analyze one uploaded pass rep from an ordered landmark sequence."""
    frames = list(frames_landmarks)
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
    contact_center = _detect_contact_center(rep_signal, valid_indices)
    rep = _score_rep(1, contact_center, frames, fps)
    reps = [rep] if rep is not None else []

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
        "reps": reps,
    }


def _score_rep(rep_index, frame_center, frames, fps):
    pre_contact_frames = max(1, int(round(PRE_CONTACT_SECONDS * fps)))
    post_contact_frames = max(1, int(round(POST_CONTACT_SECONDS * fps)))
    kinetic_pre_frames = max(1, int(round(KINETIC_PRE_CONTACT_SECONDS * fps)))
    kinetic_post_frames = max(1, int(round(KINETIC_POST_CONTACT_SECONDS * fps)))

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

    side_name, side = _best_visible_side(contact_window)
    measurements = _measure_rep(rep_window, contact_window, side)
    stability = _score_stability(measurements)
    integrity = _score_integrity(measurements)
    kinetic = _score_kinetic(measurements)
    overall = _round_score(np.mean([stability, integrity, kinetic]))
    critiques = _build_critiques(measurements, stability, integrity, kinetic)

    return {
        "rep_index": rep_index,
        "frame_start": frame_start,
        "frame_center": frame_center,
        "frame_end": frame_end,
        "contact_start": contact_start,
        "contact_end": contact_end,
        "side_used": side_name,
        "scores": {
            "stability": stability,
            "integrity": integrity,
            "kinetic": kinetic,
            "overall": overall,
        },
        "measurements": measurements,
        "critiques": critiques,
    }


def _measure_rep(rep_window, contact_window, side):
    center_frame = contact_window[len(contact_window) // 2]
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
        left_ankle = landmarks[LEFT_SIDE["ankle"]]
        right_ankle = landmarks[RIGHT_SIDE["ankle"]]

        shoulder_mid = midpoint(left_shoulder, right_shoulder)
        hip_mid = midpoint(left_hip, right_hip)
        ankle_y = max(left_ankle.y, right_ankle.y)
        body_height = max(ankle_y - shoulder_mid[1], 0.001)
        shoulder_width = max(distance(left_shoulder, right_shoulder), 0.001)

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
        forearm_parallel_delta = angle_difference(left_forearm, right_forearm)

        frame_measurements.append({
            "knee_angle": knee_angle,
            "elbow_angle": elbow_angle,
            "torso_angle": torso_angle,
            "forearm_parallel_delta": forearm_parallel_delta,
            # Stability metric: rough center-of-gravity depth. Higher values
            # mean the hips are lower relative to the feet and shoulders.
            "cog_ratio": (ankle_y - hip_mid[1]) / body_height,
            # Extra critique metric: compares foot width to shoulder width so
            # the scorer can flag a base that is too narrow or too wide.
            "stance_width_ratio": distance(left_ankle, right_ankle) / shoulder_width,
            # Extra critique metric: large horizontal shoulder/hip separation
            # can indicate torso twist or loss of body alignment.
            "shoulder_hip_offset": abs(shoulder_mid[0] - hip_mid[0]) / shoulder_width,
        })
        nose_y_values.append(landmarks[0].y)
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
        forearm_headings.append(segment_heading(
            landmarks[side["elbow"]],
            landmarks[side["wrist"]],
        ))

    center_shoulder_mid = midpoint(
        center_frame[LEFT_SIDE["shoulder"]],
        center_frame[RIGHT_SIDE["shoulder"]],
    )
    center_ankle_y = max(
        center_frame[LEFT_SIDE["ankle"]].y,
        center_frame[RIGHT_SIDE["ankle"]].y,
    )
    center_body_height = max(center_ankle_y - center_shoulder_mid[1], 0.001)

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
        center_body_height,
    )
    # Kinetic metric: platform/wrists should travel with the shoulders instead
    # of whipping independently through contact.
    measurements["platform_shoulder_sync_error"] = _relative_motion_error(
        wrist_y_values,
        shoulder_y_values,
        center_body_height,
    )
    # Integrity-through-contact metric: elbows should stay locked, not flex and
    # extend rapidly as the ball arrives.
    measurements["elbow_delta"] = _safe_range(elbow_angles)
    # Platform stability metric: forearm direction should not rotate sharply
    # during the contact window.
    measurements["forearm_angle_delta"] = _angle_range(forearm_headings)
    # Stability metric: too much vertical rise means the player popped up
    # through the pass instead of staying controlled after contact.
    measurements["body_rise"] = _safe_range(rep_hip_y_values) / center_body_height
    # Extra critique metric: normalized head bob during the rep window.
    measurements["head_y_delta"] = _safe_range(nose_y_values) / center_body_height

    return {
        key: float(round(float(value), 3)) if not math.isnan(value) else None
        for key, value in measurements.items()
    }


def _score_stability(measurements):
    # Stability combines lower-body loading, body height, and torso posture.
    knee_score = _score_target_range(measurements["knee_angle"], 120, 155, 90, 180)
    cog_score = _score_target_range(measurements["cog_ratio"], 0.32, 0.52, 0.15, 0.75)
    torso_score = _score_target_range(measurements["torso_angle"], 50, 80, 25, 90)
    rise_score = _score_max_allowed(measurements["body_rise"], 0.10, 0.28)
    return _round_score(np.mean([knee_score, cog_score, torso_score, rise_score]))


def _score_integrity(measurements):
    # Integrity focuses on whether the platform is straight and even.
    elbow_score = _score_target_range(measurements["elbow_angle"], 170, 180, 135, 180)
    parallel_score = _score_max_allowed(measurements["forearm_parallel_delta"], 10, 45)
    return _round_score(np.mean([elbow_score, parallel_score]))


def _score_kinetic(measurements):
    # Kinetic rewards connected movement: shoulders/platform can rise, but
    # they should move with hips/legs instead of breaking away as arm swing.
    shoulder_sync = _score_max_allowed(
        measurements["shoulder_hip_sync_error"],
        0.05,
        0.22,
    )
    platform_sync = _score_max_allowed(
        measurements["platform_shoulder_sync_error"],
        0.04,
        0.18,
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


def _build_critiques(measurements, stability, integrity, kinetic):
    critiques = []
    elbow_angle = _measurement(measurements, "elbow_angle")
    forearm_parallel_delta = _measurement(measurements, "forearm_parallel_delta")
    knee_angle = _measurement(measurements, "knee_angle")
    torso_angle = _measurement(measurements, "torso_angle")
    stance_width_ratio = _measurement(measurements, "stance_width_ratio")
    head_y_delta = _measurement(measurements, "head_y_delta")
    shoulder_hip_offset = _measurement(measurements, "shoulder_hip_offset")
    body_rise = _measurement(measurements, "body_rise")
    shoulder_hip_sync_error = _measurement(measurements, "shoulder_hip_sync_error")
    platform_shoulder_sync_error = _measurement(
        measurements,
        "platform_shoulder_sync_error",
    )
    elbow_delta = _measurement(measurements, "elbow_delta")
    forearm_angle_delta = _measurement(measurements, "forearm_angle_delta")

    if elbow_angle >= 170:
        critiques.append("Strong elbow extension through the platform.")
    elif not math.isnan(elbow_angle):
        critiques.append("Lock the elbows more so the platform stays firm.")

    if forearm_parallel_delta > 25:
        critiques.append("Bring the forearms closer to parallel before contact.")

    if knee_angle > 160:
        critiques.append("Bend the knees more before contact to load the legs.")
    elif knee_angle < 110:
        critiques.append("Avoid dropping too deep; stay balanced and ready to extend.")

    if torso_angle > 82:
        critiques.append("Lean slightly forward instead of staying upright.")
    elif torso_angle < 40:
        critiques.append("Keep the chest from collapsing too far over the platform.")
    if body_rise > 0.28:
        critiques.append("Avoid popping up after contact; stay low and controlled.")

    if kinetic < 75:
        critiques.append("Keep the platform connected to the legs through contact.")
    if shoulder_hip_sync_error > 0.16:
        critiques.append("Shoulders rose separately from the hips; drive more from the legs.")
    if platform_shoulder_sync_error > 0.18:
        critiques.append("Platform moved independently of the shoulders during contact.")
    if elbow_delta > 30:
        critiques.append("Elbow angle changed too much at contact; keep the platform locked.")
    if forearm_angle_delta > 35:
        critiques.append("Forearms rotated through contact; hold the platform angle steady.")

    if stance_width_ratio < 1.0:
        critiques.append("Widen the stance for a more stable passing base.")
    elif stance_width_ratio > 2.8:
        critiques.append("Narrow the stance slightly so you can move after the pass.")

    if head_y_delta > 0.08:
        critiques.append("Keep the head quieter through contact to improve control.")

    if shoulder_hip_offset > 1.2:
        critiques.append("Keep shoulders and hips more connected through the pass.")

    if stability >= 85 and integrity >= 85 and kinetic >= 85:
        critiques.append("Balanced rep with stable legs, locked elbows, and quiet shoulders.")

    return critiques


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


def _detect_contact_center(rep_signal, valid_indices):
    """Pick the strongest pose-only estimate of ball contact for one-rep mode."""
    valid_values = rep_signal[valid_indices]
    return int(valid_indices[np.nanargmax(valid_values)])


def _combined_rep_signal(hip_y, platform_score, valid_indices):
    # Rep detection proxy: a rep is most likely when the athlete is low and the
    # platform is formed. Future ball tracking can replace this contact proxy.
    hip_component = _normalize_series(hip_y, valid_indices)
    platform_component = _normalize_series(platform_score, valid_indices)
    return (0.6 * hip_component) + (0.4 * platform_component)


def _platform_score_series(frames):
    # Estimates whether the passing platform is formed by checking if wrists
    # are close together and forearms are nearly parallel.
    values = np.full(len(frames), np.nan, dtype=float)
    for index, landmarks in enumerate(frames):
        if not _has_full_pose(landmarks):
            continue

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
        parallel_delta = angle_difference(left_forearm, right_forearm)
        wrist_score = _score_max_allowed(wrist_gap, 0.75, 2.0)
        parallel_score = _score_max_allowed(parallel_delta, 15, 60)
        values[index] = np.mean([wrist_score, parallel_score])
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


def _safe_range(values):
    clean_values = [
        value for value in values
        if value is not None and not math.isnan(value)
    ]
    if not clean_values:
        return float("nan")
    return float(max(clean_values) - min(clean_values))


def _relative_motion_error(primary_values, reference_values, body_height):
    primary_delta = _motion_delta(primary_values)
    reference_delta = _motion_delta(reference_values)
    if math.isnan(primary_delta) or math.isnan(reference_delta):
        return float("nan")
    return abs(primary_delta - reference_delta) / max(body_height, 0.001)


def _motion_delta(values):
    clean_values = [
        value for value in values
        if value is not None and not math.isnan(value)
    ]
    if len(clean_values) < 2:
        return float("nan")
    return float(clean_values[-1] - clean_values[0])


def _angle_range(values):
    clean_values = [
        value for value in values
        if value is not None and not math.isnan(value)
    ]
    if not clean_values:
        return float("nan")

    anchor = clean_values[0]
    relative_values = [
        (value - anchor + 180.0) % 360.0 - 180.0
        for value in clean_values
    ]
    return float(max(relative_values) - min(relative_values))


def _round_score(value):
    if value is None or math.isnan(value):
        return 0
    return int(round(float(np.clip(value, 0, 100))))
