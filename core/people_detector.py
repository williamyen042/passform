"""Find and follow every person with YOLO-pose, in MediaPipe's landmark layout.

MediaPipe will not report a person who is small next to a prominent one: on a
murphy1 frame with three people it returned one, even asked for five.
YOLO-pose returned all three, on the GPU, with tracking ids. So people are
found here and measured elsewhere.

Its keypoints are COCO-17, which has no feet and none of MediaPipe's hand or
face detail, so they are widened into MediaPipe's 33-slot layout with the
missing joints marked invisible. Everything downstream indexes landmarks the
way it always has, and the scorer's existing fallbacks handle the gaps.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from core.ball_detector import default_device


DEFAULT_MODEL_PATH = "yolov8m-pose.pt"
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IMAGE_SIZE = 960
MEDIAPIPE_LANDMARKS = 33

# COCO-17 index -> MediaPipe index. The joints the scorer measures are all
# here; what is missing is feet (29-32), hands (17-22) and most of the face.
COCO_TO_MEDIAPIPE = {
    0: 0,                      # nose
    1: 2, 2: 5,                # eyes
    3: 7, 4: 8,                # ears
    5: 11, 6: 12,              # shoulders
    7: 13, 8: 14,              # elbows
    9: 15, 10: 16,             # wrists
    11: 23, 12: 24,            # hips
    13: 25, 14: 26,            # knees
    15: 27, 16: 28,            # ankles
}


@dataclass
class Landmark:
    x: float
    y: float
    visibility: float = 0.0


@dataclass
class PersonFrame:
    frame_index: int
    pose: List[Landmark]
    # Normalized xyxy, used to crop the passer for the finer pose pass.
    box: tuple


class PeopleDetector:
    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        confidence=DEFAULT_CONFIDENCE,
        image_size=DEFAULT_IMAGE_SIZE,
        device=None,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for YOLO-pose people detection."
            ) from exc

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.image_size = image_size
        self.device = device or default_device()

    def detect(self, frame, frame_index):
        """Every person in the frame, each with a tracking id where available."""
        height, width = frame.shape[:2]
        results = self.model.track(
            frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            persist=True,
            verbose=False,
        )
        if not results:
            return {}

        result = results[0]
        keypoints = getattr(result, "keypoints", None)
        boxes = getattr(result, "boxes", None)
        if keypoints is None or boxes is None or keypoints.xy is None:
            return {}

        xy = keypoints.xy.cpu().numpy()
        conf = (
            keypoints.conf.cpu().numpy()
            if keypoints.conf is not None
            else np.ones(xy.shape[:2])
        )
        ids = (
            boxes.id.cpu().numpy().astype(int)
            if getattr(boxes, "id", None) is not None
            # No tracker ids on the first frame, so fall back to position order.
            else np.arange(len(xy))
        )
        xyxy = boxes.xyxy.cpu().numpy()

        people = {}
        for person, (points, scores, box) in enumerate(zip(xy, conf, xyxy)):
            people[int(ids[person])] = PersonFrame(
                frame_index=frame_index,
                pose=to_mediapipe(points, scores, width, height),
                box=(
                    box[0] / width, box[1] / height,
                    box[2] / width, box[3] / height,
                ),
            )
        return people


def to_mediapipe(points, scores, width, height):
    """Widen COCO-17 keypoints into MediaPipe's 33-slot layout."""
    pose = [Landmark(0.0, 0.0, 0.0) for _ in range(MEDIAPIPE_LANDMARKS)]
    for coco_index, mediapipe_index in COCO_TO_MEDIAPIPE.items():
        x, y = points[coco_index]
        pose[mediapipe_index] = Landmark(
            float(x) / width,
            float(y) / height,
            float(scores[coco_index]),
        )

    # Feet are the one gap the scorer notices: it builds the support base from
    # 27-32 and falls back to the ankles alone when fewer than two are visible,
    # which is exactly what leaving these at zero visibility triggers.
    return pose
