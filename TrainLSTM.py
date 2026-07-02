import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler

from concurrent.futures import ThreadPoolExecutor

from sklearn.metrics import f1_score

import multiprocessing

import json


NETWORK_NAME = "LSTM_f1_03"

ROOT_DIR = "Data/Landmark"

INPUT_SIZE = 132
HIDDEN_SIZE = 64
NUM_LAYERS = 2
NUM_CLASSES = 2
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 500

WEIGHT = 0.75

hiper_params = {
    "input": INPUT_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "num_layer": NUM_LAYERS
}


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

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_labels = [train_dataset.dataset.labels[i].item() for i in train_dataset.indices]

    count_adl = train_labels.count(0)
    count_falling = train_labels.count(1)

    weight_per_class = [1.0 / count_adl, 1.0 / count_falling]
    sample_weights = [weight_per_class[label] for label in train_labels]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, sampler=sampler)

    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = FallDetectionLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    #model = torch.compile(model)

    class_weights = torch.tensor([1.0, WEIGHT], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    all_time_high_f1 = 0

    best_model_path = f"Models/{NETWORK_NAME}.pt"

    epoch = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad(set_to_none=True) # Clear previous gradients
            loss.backward()        # Compute gradients
            optimizer.step()       # Update weights
            
            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = (correct_train / total_train) * 100

        model.eval()
        val_loss = 0.0

        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.append(predicted)
                all_labels.append(labels)
                
        all_preds = torch.cat(all_preds).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100

        f1_scores = f1_score(all_labels, all_preds, average=None, zero_division=0)

        f1_adl = f1_scores[0] * 100
        f1_falling = f1_scores[1] * 100

        print(f"Epoch [{epoch+1}] "
          f"| Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% "
          f"| Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%"
          f"| F1 ADL: {f1_adl:.2f}% Falling: {f1_falling:.2f}%",end="")

        if (f1_falling+f1_adl)/2 > all_time_high_f1:
            all_time_high_f1 = (f1_falling+f1_adl)/2

            checkpoint_data = {
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'all_time_high_f1': (f1_falling+f1_adl)/2
            }

            torch.save(checkpoint_data, best_model_path)

            print(f"| Saved: {(f1_falling+f1_adl)/2:.2f}%")
        else:
            print("\n",end="")