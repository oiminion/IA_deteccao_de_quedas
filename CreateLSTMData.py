import pandas as pd

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from concurrent.futures import ProcessPoolExecutor

import os

from BuildTimeFrames import getFallingTimeFrame

def load_fall_periods(csv_path):
    # Carrega o CSV do dataset
    df = pd.read_csv(csv_path)
    
    fall_periods = {}
    
    for index, row in df.iterrows():
        scenario_id = int(row['chute'])
        camera_id = int(row['cam'])
        start_frame = int(row['start'])
        end_frame = int(row['end'])
        
        video_name = f"chute{str(scenario_id).zfill(2)}cam{camera_id}"
        
        fall_periods[video_name] = (start_frame, end_frame)
        
    return fall_periods

def process_single_video(video_info):
    video_path, video_name, fall_periods = video_info
    
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False
    )
    local_detector = vision.PoseLandmarker.create_from_options(options)
    
    # Chama a sua função original passando o detector local do processo
    try:
        getFallingTimeFrame(video_path, local_detector, video_name, fall_periods)
        return f"Sucesso: {video_name}"
    except Exception as e:
        return f"Erro no vídeo {video_name}: {str(e)}"

if __name__ == '__main__':
    CSV_PATH = "Data/data_tuple3.csv"
    fall_periods = load_fall_periods(CSV_PATH)
    dataset_dir = "Data/dataset"
    
    tarefas = []
    lista_videos = os.listdir(dataset_dir)
    total_videos = len(lista_videos)
    
    for video in lista_videos:
        video_path = os.path.join(dataset_dir, video)
        tarefas.append((video_path, video, fall_periods))

    count = 0

    print("Iniciou")

    with ProcessPoolExecutor(max_workers=max(1,os.cpu_count()-1)) as executor:
        for resultado in executor.map(process_single_video, tarefas):
            count += 1
            print(f"[{count}/{total_videos}] {count/total_videos:.2%} - {resultado}")
'''
else:
    dataset_dir = "Data/dataset"
    for video in os.listdir(dataset_dir):
        CSV_PATH = "Data/data_tuple3.csv"
        fall_periods = load_fall_periods(CSV_PATH)
        process_single_video((os.path.join(dataset_dir, video), video, fall_periods))
        break
'''