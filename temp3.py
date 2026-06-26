import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import vision


import os

def clearLandmarks(landmarks):
    result = []
    for i in range(len(landmarks)):
        if (i >= 1 and i <= 6) or (i >= 9 and i <= 10) or i == 21 or i == 22:
            continue
        result.append([landmarks[i].x, landmarks[i].y, landmarks[i].z])
    return result

base_options = python.BaseOptions(model_asset_path="E:/MediaPipe/pose_landmarker_full.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
count = 0
for file_name in os.listdir("E:/Unesp/ICD/Codigo/Datasets/Frames"):
    image = mp.Image.create_from_file(f"E:/Unesp/ICD/Codigo/Datasets/Frames/{file_name}")
    detection_result = detector.detect(image)
    print(detection_result)
    print("\n\n")
    print(detection_result.pose_landmarks[0])
    print("\n\n")
    clean = clearLandmarks(detection_result.pose_landmarks[0])
    print(len(detection_result.pose_landmarks[0]))
    print(clean)
    a = [
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
    print(len(a))
    break

[[], 
    [
        [
            [0.3952629566192627, 0.21459978818893433, -0.1907242089509964]
        ],
        [
            [0.38893255591392517, 0.22968551516532898, -0.13909703493118286]
        ]
    ], 
    [
        [
            [0.3106350898742676,0.33375853300094604, -0.1211075559258461]
        ], 
        [
            [0.2971891760826111, 0.3261396586894989, 0.03117673099040985]
        ]
    ], 
    [
        [
            [0, 0, 0]
        ], 
        [
            [0.66533362865448, 0.623643159866333, -0.15752193331718445]
        ]
    ]
]