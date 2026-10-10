import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("🎭 Calavera AR - Día de Muertos")
st.write("Permite el acceso a tu cámara para ver la máscara interactiva en tiempo real.")

# Usar el clasificador de rostros integrado de OpenCV (viene incluido sin errores externos)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class CalaveraTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Voltear horizontalmente para efecto espejo natural
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            # Coordenadas aproximadas basadas en la posición del rostro detectado
            center_x = x + w // 2
            center_y = y + h // 2
            
            # 1. Cuencas oculares (Izquierda y Derecha)
            left_eye = (center_x - int(w * 0.18), y + int(h * 0.38))
            right_eye = (center_x + int(w * 0.18), y + int(h * 0.38))
            
            cv2.circle(img, left_eye, int(w * 0.1), (20, 20, 20), -1)
            cv2.circle(img, left_eye, int(w * 0.1), (0, 255, 255), 2) # Anillo amarillo
            
            cv2.circle(img, right_eye, int(w * 0.1), (20, 20, 20), -1)
            cv2.circle(img, right_eye, int(w * 0.1), (0, 255, 255), 2)

            # 2. Nariz de calavera (Triángulo invertido)
            nose_top = (center_x, y + int(h * 0.48))
            nose_left = (center_x - int(w * 0.08), y + int(h * 0.62))
            nose_right = (center_x + int(w * 0.08), y + int(h * 0.62))
            pts_nose = np.array([nose_top, nose_left, nose_right], np.int32)
            
            cv2.fillPoly(img, [pts_nose], (20, 20, 20))
            cv2.polylines(img, [pts_nose], True, (255, 0, 127), 2) # Borde fucsia

            # 3. Flor decorativa tradicional en la frente
            forehead = (center_x, y + int(h * 0.15))
            cv2.circle(img, forehead, int(w * 0.08), (0, 128, 255), -1)
            cv2.circle(img, forehead, int(w * 0.08), (255, 255, 255), 2)

        return img

webrtc_streamer(key="calavera", video_transformer_factory=CalaveraTransformer)
