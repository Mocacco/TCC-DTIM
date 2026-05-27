import os
from ultralytics import YOLO

# 1. Carrega o seu modelo treinado (ajuste para a sua pasta real runs/classify/train-X)
caminho_modelo = "runs/classify/train-6/weights/best.pt"
model = YOLO(caminho_modelo)

# 2. Pasta onde estão as imagens de validação saudáveis
pasta_val_saudavel = "Datasets_YOLO/val/saudavel"

# 3. Filtra apenas os arquivos que pertencem ao paciente 501
id_paciente = "notumor-530"
lista_fatias = [
    os.path.join(pasta_val_saudavel, f) 
    for f in os.listdir(pasta_val_saudavel) 
    if f.startswith(id_paciente) and f.endswith(".jpg")
]

print(f"Encontradas {len(lista_fatias)} fatias para o paciente {id_paciente}.")

if len(lista_fatias) == 0:
    print("Nenhuma imagem encontrada. Verifique se o nome do paciente ou o caminho estão corretos.")
else:
    print("Iniciando a verificação de todos os slices na RTX 3060...")
    
    # 4. Executa a predição em lote (passando a lista de caminhos)
    # save=True salva as imagens com o diagnóstico desenhado no topo
    resultados = model.predict(source=lista_fatias, save=True, conf=0.5, verbose=False)
    
    # 5. Contadores para o relatório do TCC
    total_saudavel = 0
    total_tumor = 0
    
    for r in resultados:
        # Pega a classe com maior probabilidade na fatia atual
        classe_predita_idx = r.probs.top1
        classe_nome = r.names[classe_predita_idx]
        
        if classe_nome == "saudavel":
            total_saudavel += 1
        else:
            total_tumor += 1
            
    # 6. Exibe o laudo consolidado do paciente
    print("\n================ LAUDO CONSOLIDADO ================")
    print(f"Paciente ID: {id_paciente}")
    print(f"Total de fatias analisadas: {len(lista_fatias)}")
    print(f"Fatias classificadas como SAUDÁVEL: {total_saudavel}")
    print(f"Fatias classificadas como TUMOR: {total_tumor}")
    print("====================================================")
    
    # Métrica de consistência para o seu texto do TCC
    if id_paciente.startswith("tumor"):
        taxa_acerto = (total_tumor / len(lista_fatias)) * 100
    else:
        taxa_acerto = (total_saudavel / len(lista_fatias)) * 100

    print(f"Consistência do modelo para este paciente: {taxa_acerto:.2f}%")