import cv2
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision

import random

CHANCE = 0.05

def clearLandmarks(landmarks):
    result = []
    for i in range(len(landmarks)):
        if (i >= 1 and i <= 6) or (i >= 9 and i <= 10) or i == 21 or i == 22:
            continue
        result.append([landmarks[i].x, landmarks[i].y, landmarks[i].z])
    return result

def extractDataAllFrames(video_path, detector, file_name):
    camera = cv2.VideoCapture(video_path)
    count = 0
    remaining = 0
    blocks = 0
    result = []
    while True:
        success, frame = camera.read()
        if not success: break
        if remaining == 0 and random.random() < CHANCE:
            remaining = 10
        if count % 5 == 0 and remaining > 0:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = detector.detect(mp_image)
            if len(detection_result.pose_landmarks) > 0:
                detection_result = clearLandmarks(detection_result.pose_landmarks[0])
            else:
                detection_result = [
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [0,0,0]
                    ]
            
            result.append(detection_result)

            remaining -= 1

            if remaining == 0:
                numpy_array = np.array(result)
                result = []
                try:
                    name = file_name.replace(".mp4","")
                    np.save( f"Datasets/Oficial/Treino/{name}_{blocks:03d}",numpy_array)
                    blocks += 1
                except Exception as e:
                    print(e)
        count += 1
    camera.release()

# Usage

base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
count = 0

for file_name in os.listdir("Datasets/GMDCSA24_Modified/ADL"):#GMDCSA24_Modified/ADL
    extractDataAllFrames(f"Datasets/GMDCSA24_Modified/ADL/{file_name}", detector, file_name)
    
    #break