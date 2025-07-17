import cv2
import numpy as np
import speech_recognition as sr
import threading
import time
import queue

class RealTimeVoiceCamera:
    def __init__(self):
        self.running = True
        self.text_queue = queue.Queue()
        self.current_text = ""
        self.text_color = (0, 255, 0)  # Verde para mejor visibilidad
        self.font_scale = 1.3
        self.last_update_time = time.time()
        
        # Configuración optimizada de la cámara
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Máximo FPS posible
        
        # Configuración optimizada del reconocimiento de voz
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 0.5  # Menos pausa requerida
        self.recognizer.phrase_threshold = 0.1  # Detecta frases más cortas
        self.microphone = sr.Microphone()
        
    def audio_capture(self):
        """Hilo para captura continua de audio con buffer"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("Sistema de escucha activado (di 'parar' para terminar)")
            
            while self.running:
                try:
                    # Grabación más corta y sensible
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=2)
                    text = self.recognizer.recognize_google(audio, language="es-ES").lower()
                    
                    if "parar" in text:
                        self.running = False
                    elif text:
                        self.text_queue.put(text)
                        print(f"Detectado: {text}")  # Feedback inmediato
                        
                except sr.WaitTimeoutError:
                    continue  # Silencios normales, no imprimir errores
                except sr.UnknownValueError:
                    continue  # No imprimir errores de audio no reconocido
                except Exception as e:
                    print(f"Error de audio: {str(e)}")
                    time.sleep(0.1)

    def update_text(self):
        """Actualiza el texto actual desde la cola (sin bloqueo)"""
        try:
            while True:
                self.current_text = self.text_queue.get_nowait()
                self.last_update_time = time.time()
        except queue.Empty:
            pass
        
        # Borrar texto después de 5 segundos sin actualización
        if time.time() - self.last_update_time > 5:
            self.current_text = ""

    def run(self):
        # Iniciar hilo de audio
        audio_thread = threading.Thread(target=self.audio_capture, daemon=True)
        audio_thread.start()
        
        try:
            while self.running:
                # Captura de cámara optimizada
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                # Actualización del texto (no bloqueante)
                self.update_text()
                
                # Mostrar texto si existe
                if self.current_text:
                    cv2.putText(frame, self.current_text, (30, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                               self.text_color, 3, cv2.LINE_AA)
                
                # Mostrar FPS (opcional)
                cv2.putText(frame, f"FPS: {int(self.cap.get(cv2.CAP_PROP_FPS))}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           (255, 255, 255), 2)
                
                cv2.imshow('Voz en Tiempo Real', frame)
                
                # Controles rápidos
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('c'):
                    self.text_color = (np.random.randint(150, 255),
                                      np.random.randint(150, 255),
                                      np.random.randint(150, 255))
                
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            self.cap.release()
            cv2.destroyAllWindows()
            print("Sistema terminado")

if __name__ == "__main__":
    app = RealTimeVoiceCamera()
    app.run()