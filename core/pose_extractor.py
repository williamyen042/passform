#setup for mediapipe
import mediapipe as mp
import cv2
import numpy as np

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_heavy.task"),
    running_mode=VisionRunningMode.VIDEO
)

# 1. open video with cv2.VideoCapture
# 2. loop frames
# 3. convert each frame to mp.Image
# 4. run landmarker.detect_for_video(frame, timestamp)
# 5. pass landmarks to angle_calculator.py
