import cv2
import torch
from ultralytics import YOLO
import os
from datetime import datetime

# --- Configuración Inicial ---
model = YOLO("yolo12n.pt")
cap = cv2.VideoCapture(1)

# Usar GPU si está disponible
device = 0 if torch.cuda.is_available() else "cpu"
model.to(device)

if device == 0:
    print("CUDA está disponible. El modelo se ejecutará en la GPU.")
else:
    print("CUDA no está disponible. El modelo se ejecutará en la CPU.")

# --- Ruta para guardar imágenes ---
output_folder = r"C:/Users/Edzul/OneDrive/Desktop/clip/imagenes/personas_detectadas"
os.makedirs(output_folder, exist_ok=True)

# --- Conjunto para IDs ya vistos ---
seen_ids = set()

# --- Bucle Principal ---
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detección y Rastreo
    results = model.track(frame, conf=0.50, classes=[0], persist=True)
    
    # Procesar los resultados del rastreo
    if results[0].boxes.id is not None:
        for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
            track_id_int = int(track_id)
            
            # Si es un nuevo ID
            if track_id_int not in seen_ids:
                seen_ids.add(track_id_int)
                
                # Obtener la fecha y hora actuales
                current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                # Extraer y ampliar el recorte
                x1, y1, x2, y2 = map(int, box)
                
                # Ampliar en 20 píxeles por lado
                padding = 20
                x1_padded = max(0, x1 - padding)
                y1_padded = max(0, y1 - padding)
                x2_padded = min(frame.shape[1], x2 + padding)
                y2_padded = min(frame.shape[0], y2 + padding)
                
                # Recortar la imagen ampliada
                person_crop = frame[y1_padded:y2_padded, x1_padded:x2_padded]
                
                # Guardar la imagen en formato JPEG con fecha, hora e ID
                filename = os.path.join(output_folder, f"persona_{track_id_int}_{current_time}.jpeg")
                cv2.imwrite(filename, person_crop)
                print(f"Nueva persona detectada y guardada en: {filename}")

    # Mostrar resultados en la ventana
    annotated_frame = results[0].plot()
    cv2.imshow('YOLO Tracking', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

# --- Liberar recursos ---
cap.release()
cv2.destroyAllWindows()