#setup for mediapipe
import mediapipe as mp
import cv2
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
    def __init__(self, mode = "video", model_path = "pose_landmarker_heavy.task"):
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



    
