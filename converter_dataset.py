import os
import numpy as np
import cv2
from tqdm import tqdm

# Configuração dos caminhos baseada na sua estrutura atual
PASTA_VAL_ORIGEM = "Datasets/val"          # Onde estão os seus .npy de validação
PASTA_VAL_DESTINO = "Datasets_YOLO/val"    # Onde vamos salvar as fotos .jpg de validação

def converter_apenas_val(classe_nome):
    origem_completa = os.path.join(PASTA_VAL_ORIGEM, classe_nome)
    destino_completo = os.path.join(PASTA_VAL_DESTINO, classe_nome)
    
    if not os.path.exists(origem_completa):
        print(f"Aviso: Pasta {origem_completa} não encontrada.")
        return

    os.makedirs(destino_completo, exist_ok=True)
    arquivos = [f for f in os.listdir(origem_completa) if f.endswith('.npy')]
    
    print(f"\nConvertendo fatias de validação para a classe: '{classe_nome}'...")
    for arq in tqdm(arquivos):
        volume = np.load(os.path.join(origem_completa, arq))
        nome_base = os.path.splitext(arq)[0]
        
        # Coleta as fatias centrais (eixo Z de 80 a 180) para garantir imagens com estruturas visíveis
        for z in range(80, 180):
            fatia = volume[:, :, z]
            if fatia.max() > 0:  # Ignora fatias completamente pretas
                # Normaliza para o padrão de imagem de 8-bits (0-255)
                img_norm = cv2.normalize(fatia, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                # Salva no formato JPG esperado pelo YOLOv8
                nome_img = f"{nome_base}_z{z}.jpg"
                cv2.imwrite(os.path.join(destino_completo, nome_img), img_norm)

# Executa a conversão para ambas as subpastas da validação
converter_apenas_val("saudavel")
converter_apenas_val("tumor")

print("\nValidação convertida com sucesso! Agora a pasta 'Datasets_YOLO/val' está pronta.")