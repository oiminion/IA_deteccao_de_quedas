import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from pathlib import Path

dataset_paths = [
        "E:/Unesp/TCC/Codigo/Datasets/VLibrasil/videos_UFPE_(V-LIBRASIL)/data",
        "E:/Unesp/TCC/Codigo/Datasets/MINDS_Libras",
        "E:/Unesp/TCC/Codigo/Datasets/UFV"
    ]

count = 0

for string_path in dataset_paths:
        
    directory_path = Path(string_path)

    files = [item for item in directory_path.iterdir() if item.is_file()]

    for file in files:
        file_name = file.name
        label = file_name.split("@")[0]
        folder_path = Path(f"E:/Unesp/TCC/Codigo/Treino/Dados/D3_Train/{label}")
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)
    