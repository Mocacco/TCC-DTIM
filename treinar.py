import os
from ultralytics import YOLO

def iniciar_treino():
    # 1. Carrega o modelo de classificação
    model = YOLO('yolov8n-cls.pt') 

    # 2. Define o caminho absoluto para a nova pasta Datasets_YOLO
    caminho_projeto = os.path.abspath(os.path.dirname(__file__))
    caminho_dataset = os.path.join(caminho_projeto, 'Datasets_YOLO')

    print(f"Iniciando treinamento na GPU com o dataset: {caminho_dataset}")

    # 3. Executa o treinamento protegido
    results = model.train(
        data= "Datasets_YOLO",  
        epochs=50,            
        imgsz=256,            
        device=0,
        batch=64,             # Usa a sua RTX 3060
        workers=4             # Define um limite seguro de subprocessos para o Windows
    )

# Este bloco abaixo é o que resolve o erro do Multiprocessing no Windows
if __name__ == '__main__':
    iniciar_treino()