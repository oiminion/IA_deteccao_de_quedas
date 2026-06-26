
import numpy as np
import cv2
import mediapipe as mp
import random

import os

from pathlib import Path



CHANCE = 0.05

count = 0
while count < int(249/100*5):
    for file_name in os.listdir("E:/Unesp/ICD/Codigo/Datasets/Oficial/Treino"):
        if random.random() < CHANCE:
            # Define paths
            #source = Path(f"E:/Unesp/ICD/Codigo/Datasets/Oficial/Treino/{file_name}")
            #destination = Path(f"E:/Unesp/ICD/Codigo/Oficial/Teste/{file_name}")

            # Move the file (destination must include the file name)
            #source.rename(destination)
            print(file_name)
            count += 1
        if count >= int(249/100*5):
            break