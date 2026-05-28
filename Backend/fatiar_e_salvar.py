import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

PASTA_ORIGEM = "dataset"
PASTA_DESTINO = "datasets_yolo_temp"

# 1. Mapeia e separa os volumes .npy originais
arquivos = [
    os.path.join(PASTA_ORIGEM, f) 
    for f in os.listdir(PASTA_ORIGEM) 
    if f.endswith(".npy")
]

print(f"Total de volumes .npy originais localizados: {len(arquivos)}")

arquivos_train, arquivos_val = train_test_split(
    arquivos, 
    test_size=0.2, 
    random_state=42, 
    shuffle=True
)

def exportar_dataset_2d(lista_volumes, subpasta):
    print(f"\n[Fatiando] Processando conjunto de {subpasta.upper()}...")
    cont_tumor = 0
    cont_saudavel = 0

    for idx_vol, arquivo in enumerate(lista_volumes):
        nome_arquivo = os.path.basename(arquivo)
        nome_base = os.path.splitext(nome_arquivo)[0]
        volume = np.load(arquivo)
        
        # Varre os cortes geométricos centrais (Z)
        for z in range(80, 180):
            fatia = volume[:, :, z]
            
            if fatia.max() <= 0:
                continue
            
            # Classe padrão
            classe_nome = "saudavel"
            
            # Se o paciente tem tumor, valida se ESTA fatia específica contém a lesão
            if nome_arquivo.startswith("tumor"):
                fatia_norm = cv2.normalize(fatia, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                _, thresh = cv2.threshold(fatia_norm, 180, 255, cv2.THRESH_BINARY)
                pixels_tumor = np.sum(thresh == 255)
                
                if pixels_tumor > 50:
                    classe_nome = "tumor"

            # Incrementa contadores para o relatório
            if classe_nome == "tumor":
                cont_tumor += 1
            else:
                cont_saudavel += 1

            # Cria o diretório limpo esperado pela Ultralytics
            diretorio_final = f"{PASTA_DESTINO}/{subpasta}/{classe_nome}"
            os.makedirs(diretorio_final, exist_ok=True)
            
            # Normaliza e salva em JPG de alta velocidade
            fatia_norm = cv2.normalize(fatia, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(f"{diretorio_final}/{nome_base}_z{z}.jpg", fatia_norm)
            
        # Pequeno log de progresso para a tela
        if (idx_vol + 1) % 10 == 0 or (idx_vol + 1) == len(lista_volumes):
            print(f" > Volumes processados: {idx_vol + 1}/{len(lista_volumes)}")

    print(f"➔ Concluído {subpasta}! Cortes Tumor: {cont_tumor} | Cortes Saudáveis: {cont_saudavel}")

# Executa o pipeline estrutural
exportar_dataset_2d(arquivos_train, "train")
exportar_dataset_2d(arquivos_val, "val")
print("\n[SUCESSO] Dataset preparado com sucesso para a YOLOv8!")