import cv2
import os

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import vision

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
  pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

  for pose_landmarks in pose_landmarks_list:
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=pose_landmarks,
        connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
        landmark_drawing_spec=pose_landmark_style,
        connection_drawing_spec=pose_connection_style)

  return annotated_image

def extract_all_frames(video_path, detector):
    cap = cv2.VideoCapture(video_path)
    count = 0
    flag = True
    while True:
        success, frame = cap.read()
        if not success: break
        if count % 5 == 0 and flag:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            detection_result = detector.detect(mp_image)
            annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
            cv2.imshow(f"frame {int(count/5):04d}",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()

            if key == 32 or key == 13:
                flag = False
            
        count += 1
    cap.release()
    print(f"Total: {count}")

base_options = python.BaseOptions(model_asset_path="E:/MediaPipe/pose_landmarker_full.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

#for file_name in os.listdir("E:/Unesp/ICD/Codigo/Datasets/GMDCSA24_Modified/Fall"):
#    extract_all_frames(f"E:/Unesp/ICD/Codigo/Datasets/GMDCSA24_Modified/Fall/{file_name}", detector)

string = str(input("Nome:"))

extract_all_frames(f"E:/Unesp/ICD/Codigo/Datasets/GMDCSA24_Modified/Fall/{string}.mp4", detector)