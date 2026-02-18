import cv2
import os
from datetime import datetime

class PersonTracker:
    def __init__(self, model, camera_index=1):
        self.model = model
        self.cap = cv2.VideoCapture(camera_index)
        self.seen_ids = set()
        
        # Configurar carpeta de salida
        self.output_folder = "imagenes/personas_detectadas"
        os.makedirs(self.output_folder, exist_ok=True)
    
    def run(self):
        """Ejecuta el seguimiento de personas"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Detección y Rastreo
            results = self.model.track(frame, conf=0.50, classes=[0], persist=True)
            
            # Procesar los resultados del rastreo
            if results[0].boxes.id is not None:
                for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
                    track_id_int = int(track_id)
                    
                    # Si es un nuevo ID
                    if track_id_int not in self.seen_ids:
                        self.seen_ids.add(track_id_int)
                        self._save_person_crop(frame, box, track_id_int)
            
            # Mostrar resultados
            annotated_frame = results[0].plot()
            cv2.imshow('YOLO Tracking', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        
        # Liberar recursos
        self.cap.release()
        cv2.destroyAllWindows()
    
    def _save_person_crop(self, frame, box, track_id):
        """Guarda el recorte de la persona detectada"""
        # Obtener coordenadas
        x1, y1, x2, y2 = map(int, box)
        
        # Ampliar en 20 píxeles por lado
        padding = 20
        x1_padded = max(0, x1 - padding)
        y1_padded = max(0, y1 - padding)
        x2_padded = min(frame.shape[1], x2 + padding)
        y2_padded = min(frame.shape[0], y2 + padding)
        
        # Recortar la imagen ampliada
        person_crop = frame[y1_padded:y2_padded, x1_padded:x2_padded]
        
        # Guardar imagen
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(self.output_folder, f"persona_{track_id}_{current_time}.jpeg")
        cv2.imwrite(filename, person_crop)
        print(f"Nueva persona detectada y guardada en: {filename}")