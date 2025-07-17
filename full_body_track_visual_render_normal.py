import cv2
import mediapipe as mp
import time

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,  # Corregido: "detection" (no "detection")
    min_tracking_confidence=0.5
)

# Iniciar cámara (con verificación de error)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Bucle principal
while True:
    # Leer frame
    success, frame = cap.read()
    if not success:
        print("Error: No se pudo leer el frame.")
        break

    # Convertir BGR a RGB (MediaPipe requiere RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesar el frame
    results = pose.process(frame_rgb)

    # Dibujar landmarks si se detecta un cuerpo
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        )

    # Mostrar el frame
    cv2.imshow("Full Body Tracking", frame)

    # Salir con 'ESC' o 'q'
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

    # Pequeña pausa para reducir carga de CPU
    time.sleep(0.01)

# Liberar recursos
cap.release()
cv2.destroyAllWindows()