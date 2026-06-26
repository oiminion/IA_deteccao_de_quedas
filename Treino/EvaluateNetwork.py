import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_curve, roc_auc_score

import json

NETWORK_NAME = "CNN_3D_Main_01"

class AllData(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = []
        self.labels = []
        
        aux2 = 0
        for file_name in os.listdir(data_dir):
            if file_name.endswith('.npy'):
                self.samples.append(os.path.join(data_dir, file_name))
                aux2 += 1
                
                if "_Caindo" in file_name:
                    self.labels.append(1)  # 1 = Anomaly / Positive Class
                else:
                    self.labels.append(0)
                
                #if aux2 == 11:
                #    break

        self.all_data = []
     
        print(f"Total samples: {len(self.samples)}")

        for sample_path in self.samples:
            data = np.load(sample_path)
            tensor_data = torch.from_numpy(data).float().permute(2, 0, 1)
            
            self.all_data.append(tensor_data)
        #print(f"Successfully loaded {len(self.all_data)} samples.")

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return self.all_data[idx], self.labels[idx]

class CNN_3D_Main_02(nn.Module):
    def __init__(self):
        super(CNN_3D_Main_02, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(16 * 23 * 10, 128)
        self.bn_d1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 64)
        self.bn_d2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, 32)

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

if __name__ == '__main__':
    dataset = AllData(data_dir='Datasets/Oficial/Teste')
    train_loader = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=False, persistent_workers=True,num_workers=4)

    model = CNN_3D_Main_01()

    checkpoint_path = f"Modelos/{NETWORK_NAME}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False) 

    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Epoch: {checkpoint['epoch']}")

    device = torch.device("cpu")

    model.eval()
    
    center = checkpoint['center']
    
    all_scores = []
    ground_truth_labels = []
    
    with torch.no_grad():
        for blocks, labels in train_loader:
            blocks = blocks.to(device)
            embeddings = model(blocks)
            
            distances = torch.sum((embeddings - center) ** 2, dim=1)
            
            all_scores.extend(distances.cpu().numpy())
            ground_truth_labels.extend(labels.numpy())
            
    all_scores = np.array(all_scores)
    ground_truth_labels = np.array(ground_truth_labels)

    fpr, tpr, thresholds = roc_curve(ground_truth_labels, all_scores)
    roc_auc = roc_auc_score(ground_truth_labels, all_scores)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    print("\n--- Evaluation Metrics ---")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"threshold: {optimal_threshold}")

    

