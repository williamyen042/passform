"""Read the passer's pose with RTMPose, as an alternative to MediaPipe.

Same job as core/pose_extractor.py and the same output, so the two can be
swapped and compared: landmarks in MediaPipe's 33-slot layout, normalized to
the whole frame, with a visibility per joint.

Why it might be better here. MediaPipe's BlazePose is built for one person
filling a phone camera, and on this footage it is fed a crop of somebody
thirty metres away, which is where its single-frame dropouts come from - an
elbow reading 180, 118, 158 degrees across 33ms is not an arm moving, it is a
tracker losing a joint. RTMPose is top-down with a SimCC head and was trained
on multi-person scenes at this kind of scale.

Whether it is actually better on our clips is a question for
scripts/compare_pose.py, not for this docstring.

Halpe-26 rather than COCO-17 because it has feet, and the balance measurement
reads them. Models download themselves on first use, to ~/.cache/rtmlib.
"""

from dataclasses import dataclass

import numpy as np

MEDIAPIPE_LANDMARKS = 33
DEFAULT_MODE = "performance"

# Halpe-26 index -> MediaPipe index. The joints the scorer measures are all
# here, and unlike YOLO-pose's COCO-17 the feet are too: heels and big toes
# land in 29-32, which is what _projected_balance_offset reads.
HALPE_TO_MEDIAPIPE = {
    0: 0,                      # nose
    1: 2, 2: 5,                # eyes
    3: 7, 4: 8,                # ears
    5: 11, 6: 12,              # shoulders
    7: 13, 8: 14,              # elbows
    9: 15, 10: 16,             # wrists
    11: 23, 12: 24,            # hips
    13: 25, 14: 26,            # knees
    15: 27, 16: 28,            # ankles
    20: 31, 21: 32,            # big toes
    24: 29, 25: 30,            # heels
}

MODELS = {
    "lightweight": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.zip",
        (192, 256),
    ),
    "balanced": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
        (192, 256),
    ),
    "performance": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip",
        (288, 384),
    ),
}


@dataclass
class Landmark:
    """Matches what the MediaPipe path hands downstream: normalized, with a
    visibility the scorer already knows how to gate on."""
    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.0


class RTMExtractor:
    def __init__(self, mode=DEFAULT_MODE, device=None):
        try:
            from rtmlib import RTMPose
        except ImportError as exc:
            raise RuntimeError(
                "rtmlib is required for RTMPose. pip install rtmlib"
            ) from exc
        from core.ball_detector import default_device

        url, input_size = MODELS[mode]
        self.mode = mode
        # onnxruntime rather than torch: the weights are ONNX and the project
        # already runs VballNet through the same runtime.
        self.model = RTMPose(
            onnx_model=url,
            model_input_size=input_size,
            backend="onnxruntime",
            device=device or default_device(),
        )

    def landmarks_for_box(self, frame, box):
        """Pose for one person, given their normalized xyxy box.

        Top-down, using the box the people tracker already produced, so this
        never runs a second person detector and never has to be told which of
        nine people to measure.
        """
        height, width = frame.shape[:2]
        pixels = [box[0] * width, box[1] * height, box[2] * width, box[3] * height]
        keypoints, scores = self.model(frame, bboxes=[pixels])
        if keypoints is None or len(keypoints) == 0:
            return None
        return to_mediapipe_layout(keypoints[0], scores[0], width, height)


def to_mediapipe_layout(keypoints, scores, width, height):
    """Halpe-26 pixels -> 33 normalized MediaPipe slots, gaps marked invisible.

    Slots with no Halpe equivalent keep visibility 0, which is exactly how the
    YOLO-pose path already widens COCO-17, so the scorer's existing fallbacks
    handle them unchanged.
    """
    landmarks = [Landmark(0.0, 0.0, 0.0, 0.0) for _ in range(MEDIAPIPE_LANDMARKS)]
    for source, target in HALPE_TO_MEDIAPIPE.items():
        if source >= len(keypoints):
            continue
        x, y = keypoints[source]
        landmarks[target] = Landmark(
            x=float(x) / max(width, 1),
            y=float(y) / max(height, 1),
            z=0.0,
            visibility=float(scores[source]),
        )
    return landmarks
