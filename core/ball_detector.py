from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


# Stock COCO weights, whose "sports ball" class finds the real ball where the
# fine-tuned checkpoint cannot: that model was trained on 548 frames of one
# broadcast and fires on gym ceiling lights instead. Recall here is low, about
# one frame in ten, but the hits are genuine and the tracker only needs a
# handful of points to fit an arc. Swap back to models/volleyball_ball/best.pt
# once it is retrained on own-gym footage.
DEFAULT_MODEL_PATH = "yolov8x.pt"
# "sports ball" is the COCO name, so a stock yolov8n.pt still matches if the
# fine-tuned checkpoint is swapped out.
DEFAULT_TARGET_CLASSES = ("ball", "sports ball")
# Low because this detector is being used for recall, not precision. The
# projectile fit is what rejects the noise that comes with it.
DEFAULT_CONFIDENCE = 0.02
# Keep in sync with --imgsz in scripts/train_volleyball_ball_detector.py.
# Measured, not assumed: 1280 lost the ball entirely on one broadcast clip
# because the current checkpoint was trained at 640, and 960 is the size at
# which both that checkpoint and the COCO weights track cleanly.
DEFAULT_IMAGE_SIZE = 960
DEFAULT_MIN_BOX_AREA = 0.00002
DEFAULT_MAX_BOX_AREA = 0.03
DEFAULT_MAX_ASPECT_RATIO = 4.0


def default_device():
    """Apple Silicon GPU when there is one.

    Ultralytics picks CPU for inference unless told otherwise, and on this
    machine that measured 483 ms per frame against 99 ms on the GPU.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class BallDetection:
    frame_index: int
    center: Tuple[float, float]
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_name: str
    track_id: Optional[int] = None


class BallDetector:
    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        target_classes=DEFAULT_TARGET_CLASSES,
        confidence=DEFAULT_CONFIDENCE,
        image_size=DEFAULT_IMAGE_SIZE,
        min_box_area=DEFAULT_MIN_BOX_AREA,
        max_box_area=DEFAULT_MAX_BOX_AREA,
        max_aspect_ratio=DEFAULT_MAX_ASPECT_RATIO,
        device=None,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for YOLOv8 ball detection. "
                "Install project requirements before running main.py."
            ) from exc

        model_file = Path(model_path)
        if model_file.parent != Path(".") and not model_file.exists():
            raise FileNotFoundError(
                f"Ball model not found at {model_file}. "
                "Train it with scripts/train_volleyball_ball_detector.py first."
            )

        self.model = YOLO(model_path)
        if isinstance(target_classes, str):
            target_classes = (target_classes,)
        self.target_classes = {
            target_class.lower()
            for target_class in target_classes
        }
        self.confidence = confidence
        self.image_size = image_size
        self.min_box_area = min_box_area
        self.max_box_area = max_box_area
        self.max_aspect_ratio = max_aspect_ratio
        self.device = device or default_device()

    def detect(self, frame, frame_index):
        """Best single candidate for this frame, or None."""
        candidates = self.detect_candidates(frame, frame_index)
        return max(
            candidates,
            key=lambda detection: detection.confidence,
            default=None,
        )

    def detect_candidates(self, frame, frame_index):
        """Every candidate that survives the geometry filter.

        The tracker needs all of them: the real ball is often not the highest
        confidence box in a frame, so picking top-1 here throws away the
        detection that the motion filter would have kept.
        """
        height, width = frame.shape[:2]
        results = self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        names = getattr(results[0], "names", None) or getattr(self.model, "names", {})
        candidates = []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return []

        for box in boxes:
            class_id = _scalar(box.cls)
            class_name = str(_class_name(names, int(class_id))).lower()
            if class_name not in self.target_classes:
                continue

            confidence = float(_scalar(box.conf))
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            bbox = (
                _clip01(x1 / width),
                _clip01(y1 / height),
                _clip01(x2 / width),
                _clip01(y2 / height),
            )
            center = (
                _clip01(((x1 + x2) / 2.0) / width),
                _clip01(((y1 + y2) / 2.0) / height),
            )
            track_id = None
            if getattr(box, "id", None) is not None:
                track_id = int(_scalar(box.id))

            detection = BallDetection(
                frame_index=frame_index,
                center=center,
                bbox=bbox,
                confidence=confidence,
                class_name=class_name,
                track_id=track_id,
            )
            if self._passes_geometry_filter(detection):
                candidates.append(detection)

        return candidates

    def _passes_geometry_filter(self, detection):
        x1, y1, x2, y2 = detection.bbox
        width = max(x2 - x1, 0.0)
        height = max(y2 - y1, 0.0)
        area = width * height
        if area < self.min_box_area or area > self.max_box_area:
            return False

        aspect_ratio = max(
            width / max(height, 0.001),
            height / max(width, 0.001),
        )
        return aspect_ratio <= self.max_aspect_ratio


def _scalar(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "__len__"):
        return value[0].item() if hasattr(value[0], "item") else value[0]
    return value


def _class_name(names, class_id):
    if isinstance(names, dict):
        return names.get(class_id, class_id)
    try:
        return names[class_id]
    except (IndexError, KeyError, TypeError):
        return class_id


def _clip01(value):
    return max(0.0, min(1.0, float(value)))
