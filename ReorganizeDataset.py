import os
import shutil

# Defina o caminho para a sua pasta principal
# Se o script estiver na mesma pasta que 'dataset', deixe como está
dataset_dir = "Data/dataset/dataset"

# Percorre a pasta do dataset
for classe in os.listdir(dataset_dir):
    classe_path = os.path.join(dataset_dir, classe)
    
    # Verifica se é realmente um diretório (pasta de classe)
    if os.path.isdir(classe_path):
        
        # Percorre os vídeos dentro da pasta da classe
        for video in os.listdir(classe_path):
            video_path = os.path.join(classe_path, video)
            
            # Verifica se é um arquivo (o vídeo em si)
            if os.path.isfile(video_path):
                # Cria o novo nome: classevideo1.avi
                novo_nome = f"{classe}{video}"
                novo_caminho = os.path.join(dataset_dir, novo_nome)
                
                # Move e renomeia o arquivo para a raiz do dataset
                shutil.move(video_path, novo_caminho)

        try:
            os.rmdir(classe_path)
        except OSError:
            print(f"Não foi possível remover a pasta {classe_path} (pode não estar vazia)")

