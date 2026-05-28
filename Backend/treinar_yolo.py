import os
from ultralytics import YOLO

def iniciar_treinamento():
    print("\nInicializando arquitetura YOLOv8-cls...")
    # Carrega o modelo de classificação
    model = YOLO('yolov8n-cls.pt')

    print("\n--- Iniciando Treinamento com a Pipeline Otimizada da YOLOv8 ---")
    # Dispara o treino com os parâmetros ajustados para o seu Ryzen + RTX 3060
    results = model.train(
        data=os.path.abspath('datasets_yolo_temp'),
        epochs=50,
        imgsz=256,
        batch=64,         # Batch de 64 flui perfeito com fotos 2D em 12GB de VRAM
        device=0,         # Força o uso da GPU CUDA
        workers=4         # Usa 4 threads do seu Ryzen 5600 para carregar os JPGs
    )
    
    print("\nTreinamento concluído! Pesos salvos em runs/classify/train/weights/best.pt")

if __name__ == '__main__':
    iniciar_treinamento()