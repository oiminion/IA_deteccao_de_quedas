from pathlib import Path
import unicodedata
import re

def rename_MINDS(name: str) -> str:
    '''
        Renomeia os arquivos do dataset MINDS, para um padrão que será utilizado no projeto.

        O padrão é: nome_do_sinal@sinalizador@repeticao@origem.mp4

        Args:
            name (str): O nome original do arquivo.
        
        Returns:
            str: O nome renomeado do arquivo.
    '''
    
    # Remove acentos e caracteres especiais
    clean = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    clean = clean.replace(" ", "_")
    
    #remove a extensão do arquivo
    clean = clean.replace(".mp4", "")

    # Obtenção de cada parte que compõe o padrão
    
    # repetição
    clean, repetition = clean.split("-")

    # nome do sinal e sinalizador
    regex = r"([A-Z][a-z]+)([A-Z][a-z]+\d+)"
    parts = re.search(regex, clean)

    result = f"{parts.group(1)}@{parts.group(2)}@{repetition}@MINDS.mp4"

    print(result)

    return result

def rename_VLibrasil(name: str) -> str:
    '''
        Renomeia os arquivos do dataset VLibrasil, para um padrão que será utilizado no projeto.

        O padrão é: nome_do_sinal@sinalizador@repeticao@origem.mp4

        Args:
            name (str): O nome original do arquivo.
        
        Returns:
            str: O nome renomeado do arquivo.
    '''
    
    # Remove acentos e caracteres especiais
    clean = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    clean = clean.replace(" ", "_")
    
    #remove a extensão do arquivo
    clean = clean.replace(".mp4", "")

    # Obtenção de cada parte que compõe o padrão
    
    # repetição
    parts = clean.rsplit('_',1)

    parts[1] = parts[1].replace("Articulador", "Sinalizador0")

    result = f"{parts[0]}@{parts[1]}@1@VLibrasil.mp4"

    return result

directory_path = Path("E:/Unesp/TCC/Codigo/Datasets/MINDS_Libras")
#directory_path = Path("E:/Unesp/TCC/Codigo/Datasets/VLibrasil/videos_UFPE_(V-LIBRASIL)/data")#tive que excluir Avô e Avó
#directory_path = Path("E:/Unesp/TCC/Codigo/Temp/Videos")

files = [item for item in directory_path.iterdir() if item.is_file()]

for file in files:

    #new = rename_VLibrasil(file.name)
    new = file.name.replace("-", "@")

    new_name = directory_path / f"{new}"
    file.rename(new_name)
