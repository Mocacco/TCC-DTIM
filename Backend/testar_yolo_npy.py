import os
import cv2
import numpy as np
from ultralytics import YOLO

# 1. Carrega o modelo YOLOv8 treinado (Verifique se a pasta é a train ou train-X mais recente)
caminho_pesos = "Backend/runs/classify/train-3/weights/best.pt"
if not os.path.exists(caminho_pesos):
    print(f"Erro: Arquivo de pesos não encontrado em {caminho_pesos}")
    exit()

model = YOLO(caminho_pesos)

# 2. Caminho do arquivo .npy que você deseja testar (mude para o arquivo que quiser avaliar)
arquivo_teste = "Backend/dataset/tumor-139.npy" 

if not os.path.exists(arquivo_teste):
    print(f"Arquivo de teste não localizado: {arquivo_teste}")
    exit()

# 3. Carrega a matriz 3D volumétrica
volume = np.load(arquivo_teste)
print(f"Volume carregado: {arquivo_teste} | Formato original: {volume.shape}")

fatias_preparadas = []
indices_z = []

# 4. Aplica exatamente o mesmo fatiamento da nossa arquitetura de treino (Z: 80 a 180)
for z in range(80, 180):
    fatia = volume[:, :, z]
    
    # Ignora fatias sem sinal/pretas
    if fatia.max() > 0:
        # Normaliza estritamente para 8-bits (0-255)
        fatia_norm = cv2.normalize(fatia, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Converte para 3 canais RGB (exigência da arquitetura YOLOv8-cls)
        fatia_rgb = cv2.cvtColor(fatia_norm, cv2.COLOR_GRAY2RGB)
        
        fatias_preparadas.append(fatia_rgb)
        indices_z.append(z)

# 5. Executa a predição em lote na RTX 3060 de forma instantânea
print(f"Iniciando varredura de {len(fatias_preparadas)} fatias na GPU...")
resultados = model.predict(source=fatias_preparadas, device=0, verbose=False)

# 6. Contadores para o laudo consolidado por votação majoritária
total_saudavel = 0
total_tumor = 0

maior_confianca_tumor = -1.0
slice_mais_critico = None

for i, r in enumerate(resultados):
    classe_id = r.probs.top1
    classe_nome = r.names[classe_id]
    confianca = float(r.probs.top1conf)

    if classe_nome == "saudavel":
        total_saudavel += 1
    else:
        total_tumor += 1
        # Rastreia qual corte Z apresentou a maior certeza da patologia
        if confianca > maior_confianca_tumor:
            maior_confianca_tumor = confianca
            slice_mais_critico = indices_z[i]

# 7. Exibe o diagnóstico consolidado do paciente tridimensional
print("\n" + "="*40)
print("       LAUDO DE DIAGNÓSTICO MÉDICO")
print("="*40)
print(f"Arquivo analisado: {os.path.basename(arquivo_teste)}")
print(f"Total de cortes anatômicos avaliados: {len(fatias_preparadas)}")
print(f"Cortes normais (Saudáveis): {total_saudavel}")
print(f"Cortes com anomalias (Tumor): {total_tumor}")
print("-"*40)

# MUDANÇA CRÍTICA: Se houver QUALQUER fatia com tumor, o diagnóstico é positivo
if total_tumor > 0:
    taxa_comprometimento = (total_tumor / len(fatias_preparadas)) * 100
    print("DIAGNÓSTICO: >>> PRESENÇA DE TUMOR DETECTADA <<<")
    print(f"Taxa de comprometimento volumétrico: {taxa_comprometimento:.2f}%")
    print(f"Fatia de maior intensidade tumoral: Slice Z-{slice_mais_critico} (Certeza: {maior_confianca_tumor*100:.2f}%)")
else:
    taxa_normalidade = (total_saudavel / len(fatias_preparadas)) * 100
    print("DIAGNÓSTICO: >>> EXAME CLASSIFICADO COMO SAUDÁVEL <<<")
    print(f"Índice de normalidade tecidual: {taxa_normalidade:.2f}%")

print("="*40)