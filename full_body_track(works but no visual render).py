import cv2
import mediapipe as mp

print("Iniciando...")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

print("Abriendo cámara...")
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
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            print("¡Cuerpo detectado!")
        cv2.imshow('Debug', frame)
    except Exception as e:
        print(f"Error en frame: {e}")

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Ejecución completada")