import sys
import os

# Agregar rutas para importar módulos
sys.path.append('Yolov12_deteccion')
sys.path.append('Tracker')

from detector import init_yolo_model
from person_tracker import PersonTracker

def main():
    # Cargar modelo YOLO
    model = init_yolo_model()
    
    # Inicializar tracker
    tracker = PersonTracker(model, camera_index=1)
    
    try:
        # Ejecutar seguimiento
        tracker.run()
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()