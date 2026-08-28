#setup for mediapipe
import mediapipe as mp
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode


# 1. open video with cv2.VideoCapture
# 2. loop frames
# 3. convert each frame to mp.Image
# 4. run landmarker.detect_for_video(frame, timestamp)
# 5. pass landmarks to angle_calculator.py

class PoseExtractor: 
    def __init__(self, mode = "video", model_path = "pose_landmarker_heavy.task", num_poses = 1):
        self.mode = mode.lower()

        #chooses mode of use
        if self.mode == "video":
            running_mode = VisionRunningMode.VIDEO
        elif self.mode == "image":
            running_mode = VisionRunningMode.IMAGE
        elif self.mode == "live":
            running_mode = VisionRunningMode.LIVE_STREAM
        else:
            raise ValueError("mode must be 'video', 'image', or 'live'")
        

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=running_mode,
            num_poses=num_poses,
        )

        #create marker
        self.landmarker = PoseLandmarker.create_from_options(options)


    #convert openCV frames into mediapipe images
    def _to_mp_image(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    
    def process_frame(self, frame, timestamp_ms=None):
        mp_image = self._to_mp_image(frame)

        if self.mode == "image":
             return self.landmarker.detect(mp_image)
        elif self.mode == "video":
            if timestamp_ms is None:
                raise ValueError("timestamp_ms must be provided for video mode")
            return self.landmarker.detect_for_video(mp_image, timestamp_ms)
        elif self.mode == "live":
            if timestamp_ms is None:
                raise ValueError("timestamp_ms must be provided for live mode")
            self.landmarker.detect_async(mp_image, timestamp_ms)
            return None
        else:
            raise ValueError("Invalid mode")


    def get_landmarks(self, result):
        if not result.pose_landmarks:
            return None
        return result.pose_landmarks[0]


    #every pose in the frame, in whatever order mediapipe returned them
    def get_all_landmarks(self, result):
        if not result.pose_landmarks:
            return []
        return list(result.pose_landmarks)



    


#crop a square around a person and read their pose at full detail
def square_crop(frame, box, padding=0.18):
    """Square region around a normalized box, so aspect ratio is preserved.

    Angles are the whole point here, and a stretched crop would bend every one
    of them. Square in, square out.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    centre_x = (x1 + x2) / 2.0 * width
    centre_y = (y1 + y2) / 2.0 * height
    side = max((x2 - x1) * width, (y2 - y1) * height) * (1.0 + padding)
    side = max(side, 32.0)

    side = int(round(side))
    left = int(round(centre_x - side / 2))
    top = int(round(centre_y - side / 2))

    # Build the square directly and copy in whatever part of it is on screen.
    # Padding the whole frame first, as this did originally, allocated an
    # image larger than the source on every single frame.
    crop = np.zeros((side, side, 3), dtype=frame.dtype)
    x1 = max(0, left)
    y1 = max(0, top)
    x2 = min(width, left + side)
    y2 = min(height, top + side)
    if x2 > x1 and y2 > y1:
        crop[y1 - top:y2 - top, x1 - left:x2 - left] = frame[y1:y2, x1:x2]
    return crop, (left, top, side)


def to_frame_coordinates(landmarks, placement, frame_shape):
    """Map landmarks measured inside a crop back onto the whole frame."""
    left, top, side = placement
    height, width = frame_shape[:2]
    for landmark in landmarks:
        landmark.x = (left + landmark.x * side) / width
        landmark.y = (top + landmark.y * side) / height
    return landmarks
