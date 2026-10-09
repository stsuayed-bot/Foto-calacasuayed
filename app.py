import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("🎭 Calavera AR - Día de Muertos")
st.write("Permite el acceso a tu cámara para ver la máscara interactiva en tiempo real.")

# Inicializar MediaPipe Face Mesh fuera del procesador para eficiencia
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class CalaveraTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Voltear horizontalmente efecto espejo
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        # Procesar con MediaPipe
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye_center = np.array([
                    int(face_landmarks.landmark[33].x * w),
                    int(face_landmarks.landmark[33].y * h)
                ])
                right_eye_center = np.array([
                    int(face_landmarks.landmark[263].x * w),
                    int(face_landmarks.landmark[263].y * h)
                ])
                nose_tip = np.array([
                    int(face_landmarks.landmark[1].x * w),
                    int(face_landmarks.landmark[1].y * h)
                ])

                # Dibujar cuencas
                cv2.circle(img, tuple(left_eye_center), 28, (20, 20, 20), -1)
                cv2.circle(img, tuple(left_eye_center), 28, (0, 255, 255), 2)
                cv2.circle(img, tuple(right_eye_center), 28, (20, 20, 20), -1)
                cv2.circle(img, tuple(right_eye_center), 28, (0, 255, 255), 2)

                # Nariz
                pts_nose = np.array([
                    [nose_tip[0], nose_tip[1] - 10],
                    [nose_tip[0] - 12, nose_tip[1] + 12],
                    [nose_tip[0] + 12, nose_tip[1] + 12]
                ], np.int32)
                cv2.fillPoly(img, [pts_nose], (20, 20, 20))
                cv2.polylines(img, [pts_nose], True, (255, 0, 127), 2)

                # Frente
                forehead_center = (nose_tip[0], int(nose_tip[1] - 90))
                cv2.circle(img, forehead_center, 18, (0, 128, 255), -1)
                cv2.circle(img, forehead_center, 18, (255, 255, 255), 2)

        return img

webrtc_streamer(key="calavera", video_transformer_factory=CalaveraTransformer)
