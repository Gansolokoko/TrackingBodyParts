import cv2
import mediapipe as mp
import numpy as np

class DualPersonPoseTracker:
    def __init__(self):
        """Inicialización con manejo de errores incorporado"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Configuración robusta del modelo
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            smooth_landmarks=True
        )
        
        # Colores para cada persona
        self.colors = [(0, 255, 0), (0, 0, 255)]  # Verde y Rojo
        
        # Inicializar cámara con configuración segura
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara")
        
        # Establecer resolución estándar para mayor estabilidad
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Variables para tracking básico
        self.last_valid_frame = None

    def process_region(self, frame, x_start, x_end, person_id):
        """Procesa una región de la imagen con manejo de errores"""
        try:
            if frame is None or frame.size == 0:
                return frame
            
            h, w = frame.shape[:2]
            x_start_px = int(x_start * w)
            x_end_px = int(x_end * w)
            
            # Extraer región de interés con verificación de límites
            roi = frame[:, max(0, x_start_px):min(w, x_end_px)]
            if roi.size == 0:
                return frame
            
            # Convertir a RGB
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # Detectar pose
            results = self.pose.process(roi_rgb)
            
            if results.pose_landmarks:
                # Ajustar coordenadas al frame completo
                for landmark in results.pose_landmarks.landmark:
                    landmark.x = landmark.x * (x_end - x_start) + x_start
                
                # Dibujar landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=self.colors[person_id], thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=self.colors[person_id], thickness=2)
                )
                
                # Mostrar ID de persona
                if results.pose_landmarks.landmark:
                    nose = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
                    x, y = int(nose.x * w), int(nose.y * h)
                    cv2.putText(frame, f'Persona {person_id+1}', (x, y-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors[person_id], 2)
            
            return frame
            
        except Exception as e:
            print(f"Error al procesar región: {e}")
            return frame

    def run(self):
        """Bucle principal con recuperación de errores"""
        try:
            while True:
                # Leer frame con verificación
                ret, frame = self.cap.read()
                if not ret:
                    print("Error de frame, reintentando...")
                    time.sleep(0.1)
                    continue
                
                # Voltear horizontalmente para efecto espejo
                frame = cv2.flip(frame, 1)
                self.last_valid_frame = frame.copy()
                
                try:
                    # Procesar región izquierda (Persona 0)
                    frame = self.process_region(frame, 0.0, 0.5, 0)
                    
                    # Procesar región derecha (Persona 1)
                    frame = self.process_region(frame, 0.5, 1.0, 1)
                    
                    # Mostrar frame
                    cv2.imshow('Detección Simultánea de 2 Personas', frame)
                    
                except Exception as e:
                    print(f"Error en procesamiento: {e}")
                    if self.last_valid_frame is not None:
                        cv2.imshow('Detección Simultánea de 2 Personas', self.last_valid_frame)
                
                # Salir con 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            # Liberar recursos de forma segura
            self.cap.release()
            cv2.destroyAllWindows()
            if hasattr(self, 'pose'):
                self.pose.close()

if __name__ == "__main__":
    tracker = DualPersonPoseTracker()
    tracker.run()