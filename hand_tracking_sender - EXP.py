import cv2
import mediapipe as mp
import socket
import json
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuración de UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('localhost', 9999)

# Configuración de MediaPipe
hands = mp_hands.Hands(
    max_num_hands=8,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

def is_two_gesture(hand_landmarks):
    # Puntos de referencia de los dedos
    tip_ids = [4, 8, 12, 16, 20]  # Pulgar, índice, medio, anular, meñique
    
    # Verificar si el índice y medio están arriba (extendidos)
    # y los otros dedos están abajo (cerrados)
    fingers = []
    
    # Pulgar (comparación diferente)
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0]-1].x:
        fingers.append(1)  # Pulgar extendido
    else:
        fingers.append(0)  # Pulgar cerrado
    
    # Otros dedos
    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id]-2].y:
            fingers.append(1)  # Dedo extendido
        else:
            fingers.append(0)  # Dedo cerrado
    
    # El gesto de "2" es cuando solo el índice y medio están extendidos
    if fingers == [0, 1, 1, 0, 0] or fingers == [1, 1, 1, 0, 0]:
        return True
    return False

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue
    
    # Procesamiento de imagen
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    
    # Preparar datos para enviar
    hands_data = {"left": [], "right": []}
    
    two_gesture_detected = False
    
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_type = handedness.classification[0].label.lower()
            points = []
            
            for landmark in hand_landmarks.landmark:
                points.append({
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z
                })
            
            hands_data[hand_type] = points
            
            # Verificar gesto de "2"
            if is_two_gesture(hand_landmarks):
                two_gesture_detected = True
    
    # Si se detecta el gesto en cualquier mano, salir
    if two_gesture_detected:
        break
    
    # Enviar datos via UDP
    try:
        sock.sendto(json.dumps(hands_data).encode(), server_address)
    except Exception as e:
        print(f"Error enviando datos: {e}")
    
    # Visualización (opcional)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
sock.close()