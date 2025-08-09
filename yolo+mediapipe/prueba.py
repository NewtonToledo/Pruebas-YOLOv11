#!/usr/bin/env python3
import cv2
import numpy as np
from ultralytics import YOLO

DETECTION_CONF   = 0.3   # umbral para detecciones
KP_THRESHOLD     = 0.2   # umbral para dibujar keypoints
SMOOTHING_ALPHA  = 0.5   # peso del frame actual vs. previo

LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW,    RIGHT_ELBOW    = 7, 8
LEFT_WRIST,    RIGHT_WRIST    = 9, 10
ARMS_CONNECTIONS = [
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW,    LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW,    RIGHT_WRIST)
]
ARMS_KEYPOINTS = [
    LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST,
    RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
]

def draw_arms(img, keypoints, threshold):
    for i, j in ARMS_CONNECTIONS:
        x1, y1, s1 = keypoints[i]
        x2, y2, s2 = keypoints[j]
        if s1 > threshold and s2 > threshold:
            cv2.line(img, (int(x1),int(y1)), (int(x2),int(y2)), (255,255,255), 2)
    for idx in ARMS_KEYPOINTS:
        x, y, s = keypoints[idx]
        if s > threshold:
            cv2.circle(img, (int(x),int(y)), 5, (0,0,255), -1)

def main():
    model     = YOLO("modelos/yolo11n-pose.pt")
    cap       = cv2.VideoCapture(0)
    prev_kpts = None

    if not cap.isOpened():
        print("Error al abrir la cámara.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) inferencia
        results = model(frame, conf=DETECTION_CONF, imgsz=640, verbose=False)
        r       = results[0]

        # 2) chequea si viene vacío o None
        if r.keypoints is None:
            prev_kpts = None
            cv2.imshow("Brazos Suavizados", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
            continue

        kp_tensor = r.keypoints.data
        kp_np     = kp_tensor.cpu().numpy()
        # si no hay personas detectadas (batch=0)
        if kp_np.ndim == 3 and kp_np.shape[0] == 0:
            prev_kpts = None
            cv2.imshow("Brazos Suavizados", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
            continue

        # 3) selecciona la primera detección
        person_kpts = kp_np[0] if kp_np.ndim == 3 else kp_np

        # 4) suavizado exponencial
        if prev_kpts is None:
            smoothed_kpts = person_kpts.copy()
        else:
            smoothed_kpts = (
                SMOOTHING_ALPHA * person_kpts +
                (1 - SMOOTHING_ALPHA) * prev_kpts
            )
        prev_kpts = smoothed_kpts

        # 5) dibuja usando keypoints suavizados
        draw_arms(frame, smoothed_kpts, threshold=KP_THRESHOLD)

        # 6) muestra
        cv2.imshow("Brazos Suavizados", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
