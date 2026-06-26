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

def extractDataAllFrames(video_path, detector, file_name, fall_period):
    camera = cv2.VideoCapture(video_path)
    name = file_name.replace(".mp4","")
    count = 0
    remaining = 0
    blocks = 0
    result = []
    while True:
        success, frame = camera.read()
        if not success: break
        if count == fall_period[f"{name}"]:
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
                #print(numpy_array.shape)
                result = []
                try:
                    
                    np.save( f"Datasets/Oficial/Teste/{name}_Caindo",numpy_array)
                    break
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
fall_period = {
    "S1_01": 19, #10
    "S1_02": 8,  #10
    "S1_03": 12, #10
    "S1_04": 10, #10
    "S1_09": 4,  #7
    "S1_10": 3,  #9
    "S1_12": 11, #10
    "S1_15": 11, #10
    "S1_16": 17, #8
    "S2_01": 10, #13
    "S3_03": 9,
    "S2_05": 10, #10
    "S2_06": 17, #10
    "S2_07": 12, #9
    "S2_09": 14, #10
    "S2_10": 4,  #6
    "S2_11": 15, #5  
    "S2_12": 20, #6
    "S2_13": 8,  #8
    "S2_15": 10, #10
    "S2_16": 4,  #6
    "S2_17": 16, #7
    "S2_18": 57, #5
    "S2_19": 47, #9
    "S2_20": 46, #10
    "S2_21": 37,
    "S2_22": 7,
    "S2_23": 23,
    "S2_24": 21,
    "S2_25": 19,
    "S3_03": 8,
    "S3_08": 1,
    "S3_12": 11,
    "S3_14": 49,
    "S3_15": 3,
    "S3_16": 4,
    "S3_18": 11,
    "S3_20": 5,
    "S4_02": 8,
    "S4_04": 6,
    "S4_10": 10,
    "S4_11": 7,
    "S4_12": 8,
    "S4_14": 13
    }
for file_name in os.listdir("Datasets/GMDCSA24_Modified/Fall"):#GMDCSA24_Modified/ADL
    extractDataAllFrames(f"Datasets/GMDCSA24_Modified/Fall/{file_name}", detector, file_name, fall_period)
    
    #break