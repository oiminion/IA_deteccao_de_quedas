import json
import os
import cv2
import torch
import torch.nn as nn

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import collections

from GetSkeleton import getSkeleton


NETWORK_NAME = "LSTM_f1_81"

BEST_MODEL_PATH = f"Models/{NETWORK_NAME}"

MODEL_ASSET_PATH = "pose_landmarker_full.task"

with open(f'{BEST_MODEL_PATH}.json', 'r') as f:
    hiper_params = json.load(f)

INPUT_SIZE = hiper_params["input"]
HIDDEN_SIZE = hiper_params["hidden_size"]
NUM_LAYERS = hiper_params["num_layer"]
NUM_CLASSES = 2
CLASSES = ['ADL', 'Caiu']
SEQUENCE_LENGTH = 20

class FallDetectionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(FallDetectionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        

        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        
        out = self.fc(out[:, -1, :])
        
        return out

device = torch.device("cpu")
model = FallDetectionLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)

if os.path.exists(f"{BEST_MODEL_PATH}.pt"):
    model.load_state_dict(torch.load(f"{BEST_MODEL_PATH}.pt", map_location=device))
    print(f"Modelo carregado com sucesso de: {BEST_MODEL_PATH}")
else:
    print(f"Erro: O arquivo de pesos '{BEST_MODEL_PATH}' nao foi encontrado!")
    exit()

model.to(device)
model.eval()

base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
frame_window = collections.deque(maxlen=SEQUENCE_LENGTH)

count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    
    if count % 3 <= 1:
        count += 1
        continue
        
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    features = getSkeleton(frame_rgb, detector)
        
    frame_window.append(features)
        
    prediction_text = "Carregando janela"
    color = (255, 255, 255) # Branco
        
    if len(frame_window) == SEQUENCE_LENGTH:
        input_data = np.array(frame_window, dtype=np.float32)
        input_tensor = torch.tensor(input_data).unsqueeze(0).to(device) # Shape: [1, Seq_Len, 132]
        
        if input_data.sum() != 0:
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                    
                class_idx = predicted_class.item()
                prob_val = confidence.item() * 100
                
                prediction_text = f"{CLASSES[class_idx]} ({prob_val:.1f}%)"
                color = (0, 0, 255) if class_idx == 1 else (0, 255, 0) # Vermelho se cair, Verde se normal
                if class_idx == 1:
                    print("Caiu")
        else:
            prediction_text = f"NADA"
            color = (255, 0, 0)

        # Renderiza o texto na imagem
    cv2.putText(frame, prediction_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    cv2.imshow('Detector de Quedas - MediaPipe Tasks + LSTM', frame)
        
    if cv2.waitKey(10) & 0xFF == 13:
        break

    count += 1