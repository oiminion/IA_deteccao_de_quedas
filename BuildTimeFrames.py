import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from GetSkeleton import getSkeleton

SPACING = 3

FPS = 30
INTERVAL = 2 #seconds

DATA_FRAME_SIZE = int(FPS * INTERVAL / SPACING)

def getFallingTimeFrameGeneral(video_path, detector, file_name, fall_period):
    camera = cv2.VideoCapture(video_path)
    name = file_name.replace(".mp4","")

    remaining = 0
    count = 0

    result = []

    while True:
        success, frame = camera.read()
        
        if not success: break
        
        if count == fall_period[name]:
            remaining = 20

        if count % SPACING == 0 and remaining > 0:
            detection_result = getSkeleton(frame, detector)
            result.append(detection_result)
            remaining -= 1
            if remaining == 0:
                numpy_array = np.array(result)
                np.save( f"Data/Landmark/Falling/{name}",numpy_array)

def getFallingTimeFrame(video_path, detector, file_name, fall_periods):
    camera = cv2.VideoCapture(video_path)
    name = file_name.replace(".avi", "")

    start_fall, end_fall = fall_periods.get(name, (-1, -1))

    detected_each_frame = []
    count = 0

    while True:
        success, frame = camera.read()
        if not success: 
            break
        
        if count % 1 == 0:#1 pois o video parece acelerado
            detection_result = getSkeleton(frame, detector) 
            detected_each_frame.append((count, detection_result))
            
        count += 1
    
    camera.release()

    for i in range(0, len(detected_each_frame) - DATA_FRAME_SIZE + 1, 5):
        window = detected_each_frame[i : i + DATA_FRAME_SIZE]
        
        window_skeletons = np.array([item[1] for item in window])
        
        first_frame_idx = window[0][0]
        last_frame_idx = window[-1][0]
        
        if start_fall <= last_frame_idx and first_frame_idx <= end_fall - DATA_FRAME_SIZE:
            folder = "Data/Landmark/Falling"

        else:
            folder = "Data/Landmark/ADL"
            
        # Salva incluindo o índice da janela no nome para não sobrescrever
        np.save(f"{folder}/{name}_win_{i}", window_skeletons)




