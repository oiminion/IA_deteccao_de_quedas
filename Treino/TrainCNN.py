import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime, timedelta
import json

from pathlib import Path

from sklearn.metrics import roc_curve, roc_auc_score

NETWORK_NAME = "CNN_3D_Main_03"

class AllData2(Dataset):
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

class AllData(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = []
        
        aux2 = 0
        for file_name in os.listdir(data_dir):
            if file_name.endswith('.npy'):
                self.samples.append(os.path.join(data_dir, file_name))
                aux2 += 1
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
        # Instantly returns from RAM, zero disk I/O bottlenecks!
        return self.all_data[idx]

class CNN2D(nn.Module):
    def __init__(self):
        super(CNN2D, self).__init__()
        
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
    torch.set_num_threads(5)
    start_time = datetime.now()
    #print(f"Started: {start_time}")

    device = torch.device("cpu")

    #Carregar dataset
    dataset = AllData(data_dir=f"E:/Unesp/ICD/Codigo/Datasets/Oficial/Treino/")
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=False, persistent_workers=True,num_workers=4)#max(1,os.cpu_count()-1))

    dataset2 = AllData2(data_dir='E:/Unesp/ICD/Codigo/Datasets/Oficial/Teste')
    test_loader = DataLoader(dataset2, batch_size=32, shuffle=False, pin_memory=False, persistent_workers=True,num_workers=4)

    #Iniciar o modelo, criteriador e otimizador
    model = CNN2D()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for batch in train_loader:
            all_embeddings.append(model(batch.to(device)))
    
    center = torch.cat(all_embeddings, dim=0).mean(dim=0)
    # Avoid zero-weights mapping anomalies straight to center trivially
    center[(abs(center) < 0.1)] = 0.1 
    print("Center initialized.")

    best_auc = 0
    epoch = 0
    delta = []
    epoch_loss = 0

    #Carregar checkpoint
    checkpoint_path = f"E:/Unesp/ICD/Codigo/Modelos/{NETWORK_NAME}.pt"
    
    if Path(checkpoint_path).is_file():
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu')) 

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
        epoch_loss = checkpoint['loss']
        best_auc = checkpoint['best_auc']
        center = checkpoint.get('center', None)

        print(f"All time high: {best_auc}")

        with open(f"E:/Unesp/ICD/Codigo/Metricas/time_taken_training_{NETWORK_NAME}.json", 'r', encoding='utf-8') as file:
            # Load and parse the saved JSON data
            data = json.load(file)
            print(data)
            start_time = datetime.strptime(data["start_time"], "%Y-%m-%d %H:%M:%S.%f")
            print(f"S:{start_time}")

    #Iniciar treino

    model.train()

    flag = False
    while True:  
          
        running_loss = 0.0
        total_distance = 0.0
        all_distances = []
        total = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Map input grids to embeddings
            embeddings = model(batch)
            
            # Loss is the mean squared distance of embeddings to the center 'c'
            distances = torch.sum((embeddings - center) ** 2, dim=1)
            loss = torch.mean(torch.sum((embeddings - center) ** 2, dim=1))
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.size(0)
            total_distance += distances.sum().item()
            all_distances.extend(distances.detach().cpu().numpy())

        # Calculate Epoch Quality Metrics
        epoch_loss = running_loss / len(dataset)
        avg_radius = total_distance / len(dataset)
        radius_std = np.std(all_distances)

        all_scores = []
        ground_truth_labels = []

        model.eval()

        with torch.no_grad():
            for blocks, labels in test_loader:
                blocks = blocks.to(device)
                embeddings = model(blocks)
                
                distances = torch.sum((embeddings - center) ** 2, dim=1)
                
                all_scores.extend(distances.cpu().numpy())
                ground_truth_labels.extend(labels.numpy())

        model.train()  

        all_scores = np.array(all_scores)
        ground_truth_labels = np.array(ground_truth_labels)

        fpr, tpr, thresholds = roc_curve(ground_truth_labels, all_scores)
        epoch_auc = roc_auc_score(ground_truth_labels, all_scores)

        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        if epoch % 100 == 0:
            print(f"Epoch {epoch+1} - Loss: {epoch_loss} - radius std: {radius_std} - best auc: {best_auc} - threshold: {optimal_threshold}")

        if epoch_auc > best_auc and radius_std > 0.1:
            best_auc = epoch_auc
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'best_auc': best_auc,
                'center': center,
                'optimal_threshold': optimal_threshold
            }
            torch.save(checkpoint_data, f'E:/Unesp/ICD/Codigo/Modelos/{NETWORK_NAME}.pt')

            end_time = datetime.now()
        
            delta_time = end_time - start_time

            time_taken = {
                "start_time": str(start_time),
                "epoch": epoch,
                "best_auc": best_auc,
                "optimal_threshold": optimal_threshold,
                "time_taken": str(delta_time)
            }
            with open(f"E:/Unesp/ICD/Codigo/Metricas/time_taken_training_{NETWORK_NAME}.json", 'w') as json_file:
                json.dump(time_taken, json_file, indent=4)

        epoch += 1

    