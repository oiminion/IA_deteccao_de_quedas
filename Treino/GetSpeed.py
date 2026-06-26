import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from collections import deque

from datetime import datetime, timedelta

MIN_EPOCH = 10
MAX_EPOCH = 10000


class AllData(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = []
        self.labels = []
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(os.listdir(data_dir)))}
        
        for cls_name, cls_idx in self.class_to_idx.items():
            cls_folder = os.path.join(data_dir, cls_name)
            if os.path.isdir(cls_folder):
                for file_name in os.listdir(cls_folder):
                    if file_name.endswith('.npy'):
                        self.samples.append(os.path.join(cls_folder, file_name))
                        self.labels.append(cls_idx)

        self.all_data = []
        self.all_labels = []
        
        for sample_path, label in zip(self.samples, self.labels):
            data = np.load(sample_path)
            tensor_data = torch.from_numpy(data).float().permute(2, 0, 1)
            
            self.all_data.append(tensor_data)
            self.all_labels.append(torch.tensor(label, dtype=torch.long))
        #print(f"Successfully loaded {len(self.all_data)} samples.")

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        # Instantly returns from RAM, zero disk I/O bottlenecks!
        return self.all_data[idx], self.all_labels[idx]

class SpatioTemporalCNN(nn.Module):
    def __init__(self, num_classes):
        super(SpatioTemporalCNN, self).__init__()
        
        # Input shape: (Batch, 3, 79, 42)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Max pooling reduces spatial dimensions by half
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Let's calculate the shape after pooling steps:
        # Start: 79 x 42
        # After Pool 1: 39 x 21
        # After Pool 2: 19 x 10
        # After Pool 3: 9 x 5
        
        self.fc1 = nn.Linear(64 * 9 * 5, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Convolutional Blocks
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten for Dense Layers
        x = x.view(x.size(0), -1) 
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        # Instantly returns from RAM, zero disk I/O bottlenecks!
        return self.all_data[idx], self.all_labels[idx]

if __name__ == '__main__':
    wor = 11
    for thr in range(2,8):
        torch.set_num_threads(thr+1)
        start_time = datetime.now()
        #print(f"Started: {start_time}")

        # 1. Initialize Dataset and Loader
        dataset = AllData(data_dir='E:/Unesp/TCC/Codigo/Treino/Dados/D3')
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=False, persistent_workers=True,num_workers=wor)#max(1,os.cpu_count()-1))
        #print(f"Number of cpus: {max(1,os.cpu_count()-1)}")

        # 2. Initialize Model, Loss, and Optimizer
        num_classes = len(dataset.class_to_idx)
        model = SpatioTemporalCNN(num_classes=num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 3. Training
        epoch = 0
        delta = deque()
        past_acc = 0
        flag = False
        #while ((epoch < MIN_EPOCH) or ((sum(abs(x) for x in delta)/len(delta) >= 0.00001) and len(delta >= 50))) and (MAX_EPOCH > epoch):
        while epoch < MIN_EPOCH:
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # inputs shape: [32, 3, 79, 42]
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                # Track statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                '''
                if batch_idx == 0:
                    print(print(f"Batch successfully processed. Output shape: {outputs.shape}"))
                    break
                '''
            epoch_loss = running_loss / total
            epoch_acc = (correct / total) * 100
                    
            delta.append(epoch_acc - past_acc)

            if len(delta) >= 50:
                delta.popleft()

            past_acc = epoch_acc 

            #print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}% - Delta: {sum(abs(x) for x in delta)/len(delta)}")

            #torch.save(model.state_dict(), 'E:/Unesp/TCC/Codigo/Modelos/CNN_3D_01.pt')

            epoch += 1

            if datetime.now()-start_time >= timedelta(minutes=1,seconds=30):
                flag = True
                break

                    #print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}% - Delta: {sum(abs(x) for x in delta)/len(delta)}")

                    #torch.save(model.state_dict(), 'E:/Unesp/TCC/Codigo/Modelos/CNN_3D_01.pt')

                
        if flag:
            print(f"W: {wor} - T: {thr+1} - D: 9:99:99.999999")
            with open("E:/Unesp/TCC/Codigo/Metricas/Speed_Training.txt", "a") as file:
                file.write(f"W: {wor} - T: {thr+1} - D: 9:99:99.999999\n")
        else:
            print(f"W: {wor} - T: {thr+1} - D: {str(datetime.now()-start_time)}")
            with open("E:/Unesp/TCC/Codigo/Metricas/Speed_Training.txt", "a") as file:
                file.write(f"W: {wor} - T: {thr+1} - D: {str(datetime.now()-start_time)}\n")

    with open("E:/Unesp/TCC/Codigo/Metricas/Speed_Training.txt", "a") as file:
        file.write(f"\n")


    
    