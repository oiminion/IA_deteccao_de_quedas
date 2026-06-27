import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

VIS_THRESHOLD = 0.3
MASK_VALUE = 0

def normalizeLandmark(pose_landmarks):
    vis = np.array([lm.visibility for lm in pose_landmarks[0]])

    lms = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks[0]])
    
    left_hip = lms[23]
    right_hip = lms[24]
    hip_center = (left_hip + right_hip) / 2.0
    
    translated_lms = lms - hip_center
    
    left_shoulder = lms[11]
    right_shoulder = lms[12]
    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    
    torso_size = np.linalg.norm(shoulder_center - hip_center)
    
    if torso_size == 0:
        torso_size = 1e-6
        
    normalized_lms = translated_lms / torso_size

    normalized_lms[vis < VIS_THRESHOLD] = MASK_VALUE

    combined_lms = np.hstack((normalized_lms, vis[:, np.newaxis]))

    combined_lms[vis < VIS_THRESHOLD, 3] = MASK_VALUE

    return combined_lms.flatten()

def getSkeleton(frame, detector):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(mp_image)
    if detection_result.pose_landmarks:
        flat_normalized_features = normalizeLandmark(detection_result.pose_landmarks)
        return flat_normalized_features
    else:
        flat_normalized_features = [0 for _ in range(33*4)]
        return flat_normalized_features





