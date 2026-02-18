import torch
from ultralytics import YOLO

def load_yolo_model(model_path="yolo12n.pt"):
    """Carga el modelo YOLO y configura el dispositivo"""
    model = YOLO(model_path)
    
    # Usar GPU si está disponible
    device = 0 if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    if device == 0:
        print("CUDA está disponible. El modelo se ejecutará en la GPU.")
    else:
        print("CUDA no está disponible. El modelo se ejecutará en la CPU.")
    
    return model