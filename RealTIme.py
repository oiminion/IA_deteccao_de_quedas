import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_curve, roc_auc_score

from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision

import cv2

NETWORK_NAME = "CNN_3D_Main_01"

class CNN_3D_Main_01(nn.Module):#0.74
    def __init__(self):
        super(CNN_3D_Main_01, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(32 * 23 * 10, 128)
        self.bn_d1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 128)
        self.bn_d2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 32)

    def forward(self, x):
        #Parte convolucional
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        #Achatamento
        x = x.view(x.size(0), -1) 
        
        #Parte densa
        x = F.relu(self.bn_d1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_d2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def clearLandmarks(landmarks):
    result = []
    for i in range(len(landmarks)):
        if (i >= 1 and i <= 6) or (i >= 9 and i <= 10) or i == 21 or i == 22:
            continue
        result.append([landmarks[i].x, landmarks[i].y, landmarks[i].z])
    return result

def extractDataFromFrame(detector, block, frame):
    
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
            
    block.append(detection_result)

    if len(block) == 10:
        numpy_array = np.array(block)
        block = block[1:]
        return numpy_array, True
    else:
        return 0, False
                
def predictSingleArray(array, model, center, threshold):
    # 1. Ensure model is in evaluation mode
    model.eval()
    device = torch.device("cpu")
    
    # 2. Match the preprocessing inside your AllData Dataset class
    tensor_data = torch.from_numpy(array).float().permute(2, 0, 1).unsqueeze(0)
    tensor_data = tensor_data.to(device)
    
    # 3. Convert center to a PyTorch tensor if it isn't one
    if isinstance(center, np.ndarray):
        center = torch.from_numpy(center).float()
    center = center.to(device)
    
    # 4. Pass through model to get the embedding
    with torch.no_grad():
        embedding = model(tensor_data)
        
        # 5. Calculate squared Euclidean distance to the center
        distance = torch.sum((embedding - center) ** 2, dim=1).item()
    
    # 6. Classification decision
    is_anomaly = distance > threshold
    
    return is_anomaly, distance



#setup MediaPipe 
base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)


#setup CNN
model = CNN_3D_Main_01()

checkpoint_path = f"Modelos/{NETWORK_NAME}.pt"

epoch = -1
epoch_loss = -1
best_auc = -1
center = -1

if Path(checkpoint_path).is_file():
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False) 

    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
    epoch_loss = checkpoint['loss']
    best_auc = checkpoint['best_auc']
    center = checkpoint.get('center', None)
    threshold = checkpoint['optimal_threshold']

    print(f"All time high: {best_auc}")

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS, 30)
count = 0
time = 0
block = []
while True:
    success, frame = camera.read()
    if not success: break
    if count % 5 == 0:
        array, flag = extractDataFromFrame(detector, block, frame)
        if flag:
            anomaly_flag, dist = predictSingleArray(array, model, center, threshold)
            
            if anomaly_flag:
                print(f"Caiu -- dist: {dist}")
                time = 12
            else:
                print(f"Nada -- dist: {dist}")

            if time > 0:
                text = "CAIU"
                h, w, _ = frame.shape
                font_scale = 1
                thickness = 2
                font = cv2.FONT_HERSHEY_SIMPLEX
                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = (w - text_w) // 2
                text_y = (h + text_h) // 2
                position = (text_x, text_y)
                
                color = (255, 0, 0)
                
                line_type = cv2.LINE_AA

                cv2.putText(frame, text, position, font, font_scale, color, thickness, line_type)

                time -= 1

            cv2.imshow('Teste', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 32 or key == 13:
                break
    count += 1
camera.release()