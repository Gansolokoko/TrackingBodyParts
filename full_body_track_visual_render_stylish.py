import cv2
import mediapipe as mp
import numpy as np

# Configuración de MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Paleta de colores estilo videojuego
HUESO_COLOR = (0, 255, 255)  # Amarillo neón para huesos
ARTICULACION_COLOR = (255, 0, 255)  # Magenta neón para articulaciones
GROSOR_HUESOS = 5  # Grosor de las líneas de conexión
TAMANO_ARTICULACIONES = 8  # Tamaño de los círculos de las articulaciones

print("Iniciando seguimiento corporal estilo videojuego...")

# Inicializar modelo de pose
pose = mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=1
)

# Inicializar cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Fallo al abrir cámara. Probando con vídeo...")
    cap = cv2.VideoCapture("test_video.mp4")
    if not cap.isOpened():
        print("Error crítico: No se pudo abrir ningún dispositivo")
        exit()

print("Comenzando bucle principal...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Fin del vídeo o error de captura")
        break
    
    try:
        # Convertir a RGB (MediaPipe requiere este formato)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detección de pose
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            print("¡Cuerpo detectado! Renderizando huesos...")
            
            # Dibujar conexiones óseas (huesos)
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=ARTICULACION_COLOR,
                    thickness=TAMANO_ARTICULACIONES,
                    circle_radius=TAMANO_ARTICULACIONES
                ),
                mp_drawing.DrawingSpec(
                    color=HUESO_COLOR,
                    thickness=GROSOR_HUESOS,
                    circle_radius=0  # No dibujar círculos en las conexiones
                )
            )
            
            # Efecto adicional: brillo en articulaciones
            h, w = frame.shape[:2]
            overlay = frame.copy()
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(overlay, (x, y), TAMANO_ARTICULACIONES+5, (255, 255, 255), -1)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Mostrar FPS (opcional)
        cv2.putText(frame, "Modo: VIDEOJUEGO | Q: Salir", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Body Tracking - Estilo Videojuego', frame)
        
    except Exception as e:
        print(f"Error en frame: {e}")

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Ejecución completada")