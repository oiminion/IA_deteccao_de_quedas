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

from concurrent.futures import ThreadPoolExecutor

from torch.utils.data import Dataset, DataLoader

import multiprocessing

import torch.optim as optim

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

NETWORK_NAME = "LSTM_f1_81"

BATCH_SIZE = 64

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

ROOT_DIR = "Data/Landmark"

class LoadNumpyArray(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = ['ADL', 'Falling']
        
        self.data = []
        self.labels = []
        
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        print("Carregando o dataset para a memoria RAM")
        
        file_tasks = []
        for cls_name in self.classes:
            class_folder = os.path.join(root_dir, cls_name)
            if not os.path.exists(class_folder):
                continue
                
            for file in os.listdir(class_folder):
                if file.endswith('.npy'):
                    path = os.path.join(class_folder, file)
                    label = self.class_to_idx[cls_name]
                    file_tasks.append((path, label))
        
        with ThreadPoolExecutor(max_workers=max(1,os.cpu_count()-1)) as executor:
            # Envia todos os arquivos para serem lidos concorrentemente
            results = list(executor.map(self._load_single_file, file_tasks))
            
        # 3. Desempacota os resultados para as listas do Dataset
        for tensor_data, tensor_label in results:
            self.data.append(tensor_data)
            self.labels.append(tensor_label)
                    
        print(f"{len(self.data)} arquivos foram carregados na memoria.")

    def _load_single_file(self, task):
        path, label = task
        window = np.load(path)
        
        # Converte direto para tensor dentro da thread paralela
        tensor_data = torch.tensor(window, dtype=torch.float32)
        tensor_label = torch.tensor(label, dtype=torch.long)
        
        return tensor_data, tensor_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

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

if __name__ == '__main__':
    #torch.set_num_threads(5)
    num_cores = multiprocessing.cpu_count()-1 
    
    # Configura o PyTorch para usar o máximo de threads nas operações internas
    torch.set_num_threads(num_cores)
    torch.set_num_interop_threads(num_cores)

    device = torch.device("cpu")

    dataset = LoadNumpyArray(root_dir=ROOT_DIR)

    val_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = FallDetectionLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)

    epoch = -1

    if os.path.exists(f"{BEST_MODEL_PATH}.pt"):
        checkpoint = torch.load(f"{BEST_MODEL_PATH}.pt", map_location=device)
        model.load_state_dict(checkpoint)
        #model.load_state_dict(checkpoint['model_state_dict'])
        #epoch = checkpoint['epoch']
        print(f"Modelo carregado com sucesso de: {BEST_MODEL_PATH}")
    else:
        print(f"Erro: O arquivo de pesos '{BEST_MODEL_PATH}' nao foi encontrado!")
        exit()

    criterion = nn.CrossEntropyLoss()#weight=class_weights)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
                
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_labels.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
        
        val_avg_loss = val_loss / total
        val_avg_acc = (correct / total) * 100

        precision = 100 * precision_score(all_labels, all_predictions, average=None, zero_division=0)
        recall = 100 * recall_score(all_labels, all_predictions, average=None, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average=None, zero_division=0)

        f1_adl = f1[0] * 100
        f1_falling = f1[1] * 100

        np.save(f"Metrics/{NETWORK_NAME}_Confusion.npy", confusion_matrix(all_labels,all_predictions))
        np.save(f"Metrics/{NETWORK_NAME}_Normalized_Confusion.npy", confusion_matrix(all_labels,all_predictions,normalize='true'))

        time_taken = {
            "epoch": epoch,
            "acc": val_avg_acc,
            "precision": (precision[0]+precision[1])/2,
            "precision_ADL": precision[0],
            "precision_falling": precision[1],
            "recall": (recall[0]+recall[1])/2,
            "recall_ADL": recall[0],
            "recall_falling": recall[1],
            "f1": (f1_falling+f1_adl)/2,
            "f1_ADL": f1_adl,
            "f1_falling": f1_falling
        }
        with open(f"Metrics/{NETWORK_NAME}_info.json", 'w') as json_file:
            json.dump(time_taken, json_file, indent=4)