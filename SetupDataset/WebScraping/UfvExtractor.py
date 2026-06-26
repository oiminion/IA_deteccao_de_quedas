import requests
from bs4 import BeautifulSoup
import os
import json
from datetime import datetime

start_time = datetime.now()

# Obter as classes dos vídeos a partir do dataset
import pandas as pd

VLibrasil_df = pd.read_csv("E:/Unesp/TCC/Codigo/Metricas/data_from_video_VLibrasil.csv")
MINDS_df = pd.read_csv("E:/Unesp/TCC/Codigo/Metricas/data_from_video_MINDS.csv")

df = pd.concat([VLibrasil_df, MINDS_df], ignore_index=True)

classes = df['class'].unique()

# Obter as categorias das palavras a partir do site da UFV
url = "https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=abacaxi&term=temas&value=alimentos"

html = requests.get(url).text
soup = BeautifulSoup(html, "html.parser")

themes = []
for groups in soup.find_all("div", class_="content selectorContent temas"):
    for group in groups.find_all("a", href=True):
        src = group.get("href")
        if src:
            themes.append(src)

print(f"Total de categorias: {len(themes)}")

# Obter os links das abas dos vídeos a partir das categorias
links = []
aux = 0
flag = False
for url in themes:
    print(f"Processando categoria: {aux}/{len(themes)}")
    aux += 1
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    for divs in soup.find_all("div", id="results"):
        tags_a = divs.find_all("a", href=True)
        for tag_a in tags_a:
            src = tag_a.get("href")
            if src:
                cls = src.replace("https://sistemas.cead.ufv.br/capes/dicionario/?cadastros=", "")
                cls = cls.split('&')[0]
                cls = cls.replace("-cor", "")
                if not ("-feira" in cls or "-mail" in cls):
                    cls = cls.replace("-", "_")
                
                if cls != "e-mail":
                    cls = cls[:1].upper() + cls[1:]
            
                if cls in classes:
                    if not "configuracao_mao=configuracao" in src:
                        html2 = requests.get(src).text
                        soup2 = BeautifulSoup(html2, "html.parser")
                        for aux2 in soup2.find_all("video"):
                            src2 = aux2.get("src")
                            if src2:
                                links.append([src2, cls])

#for video in links:
#    print(video)
    
#print(f"Total de vídeos: {len(videos)}")

folder_path = "E:/Unesp/TCC/Codigo/Datasets/UFV"

for link in links:
    video_url = link[0]
    cls = link[1]
    filename = f"{cls}@Sinalizador01@1@UFV.mp4"
    if video_url.endswith(".mp4"):
        with requests.get(video_url, stream=True) as response:
            # Raise an exception for HTTP errors (404, 500, etc.)
            response.raise_for_status() 
            
            file_path = os.path.join(folder_path, filename)
            # Open the local file in "write binary" (wb) mode
            with open(file_path, 'wb') as file:
                # Write the file in 8KB chunks to preserve system RAM
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)

end_time = datetime.now()
        
delta_time = end_time - start_time

time_taken = {
    "time_taken": str(delta_time)
}
with open(f"E:/Unesp/TCC/Codigo/Metricas/time_taken_WebScraping_UFV.json", 'w') as json_file:
    json.dump(time_taken, json_file, indent=4)
#for theme in themes:
#    print(theme)
