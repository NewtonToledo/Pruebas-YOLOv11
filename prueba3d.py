#!/usr/bin/env python3
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DETECTION_CONF   = 0.3   # umbral para detecciones YOLO
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

COCO_TO_MEDIAPIPE = {
    5: 11,   # LEFT_SHOULDER
    6: 12,   # RIGHT_SHOULDER
    7: 13,   # LEFT_ELBOW
    8: 14,   # RIGHT_ELBOW
    9: 15,   # LEFT_WRIST
    10: 16,  # RIGHT_WRIST
}

mp_pose = mp.solutions.pose

fig = plt.figure(figsize=(15, 6))
ax_2d = fig.add_subplot(121)
ax_3d = fig.add_subplot(122, projection='3d')
plt.ion()
plt.tight_layout()

def normalize_coordinates(keypoints, img_width, img_height):
    """Normaliza las coordenadas de píxeles a rango [0,1]"""
    normalized = keypoints.copy()
    normalized[:, 0] = keypoints[:, 0] / img_width   
    normalized[:, 1] = keypoints[:, 1] / img_height  
    return normalized

def get_z_from_mediapipe(frame, yolo_keypoints, pose_processor):
    """Obtiene las coordenadas Z de MediaPipe para los keypoints de YOLO"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_processor.process(rgb_frame)
    
    z_coords = {}
    world_coords = {}  
    
    if results.pose_world_landmarks:
        world_landmarks = results.pose_world_landmarks.landmark
        for coco_idx, mp_idx in COCO_TO_MEDIAPIPE.items():
            if coco_idx < len(yolo_keypoints):
                z_coords[coco_idx] = world_landmarks[mp_idx].z
                world_coords[coco_idx] = {
                    'x': world_landmarks[mp_idx].x,
                    'y': world_landmarks[mp_idx].y,
                    'z': world_landmarks[mp_idx].z
                }   
    return z_coords, world_coords

def create_3d_keypoints(yolo_keypoints, z_coords, world_coords, img_width, img_height, use_hybrid=False):
    keypoints_3d = []    
    for i, (x, y, conf) in enumerate(yolo_keypoints):
        if i in COCO_TO_MEDIAPIPE and conf > KP_THRESHOLD:
            if use_hybrid:
                x_norm = x / img_width
                y_norm = y / img_height
                z = z_coords.get(i, 0.0)
            else:
                if i in world_coords:
                    x_norm = world_coords[i]['x']
                    y_norm = world_coords[i]['y'] 
                    z = world_coords[i]['z']
                else:
                    x_norm = x / img_width
                    y_norm = y / img_height
                    z = 0.0
            
            keypoints_3d.append([x_norm, y_norm, z, conf])
        else:
            keypoints_3d.append([0, 0, 0, 0]) 
    return np.array(keypoints_3d)

def draw_arms_2d(img, keypoints, threshold):
    for i, j in ARMS_CONNECTIONS:
        if i < len(keypoints) and j < len(keypoints):
            x1, y1, s1 = keypoints[i][:3]
            x2, y2, s2 = keypoints[j][:3]
            if s1 > threshold and s2 > threshold:
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
    
    for idx in ARMS_KEYPOINTS:
        if idx < len(keypoints):
            x, y, s = keypoints[idx][:3]
            if s > threshold:
                cv2.circle(img, (int(x), int(y)), 6, (0, 0, 255), -1)

def plot_2d_arms(ax, keypoints, threshold, img_width, img_height):
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0) 
    ax.set_aspect('equal')
    
    for i, j in ARMS_CONNECTIONS:
        if i < len(keypoints) and j < len(keypoints):
            x1, y1, s1 = keypoints[i][:3]
            x2, y2, s2 = keypoints[j][:3]
            if s1 > threshold and s2 > threshold:
                x1_norm = x1 / img_width
                y1_norm = y1 / img_height
                x2_norm = x2 / img_width
                y2_norm = y2 / img_height
                ax.plot([x1_norm, x2_norm], [y1_norm, y2_norm], 'g-', linewidth=2)
    
    for idx in ARMS_KEYPOINTS:
        if idx < len(keypoints) and keypoints[idx][2] > threshold:
            x, y = keypoints[idx][:2]
            x_norm = x / img_width
            y_norm = y / img_height
            ax.scatter([x_norm], [y_norm], c='red', s=50)
    
    ax.set_xlabel('X (normalizado)')
    ax.set_ylabel('Y (normalizado)')
    ax.set_title('Vista 2D - YOLO Keypoints')
    ax.grid(True, alpha=0.3)
def plot_3d_arms(ax, keypoints_3d, threshold):
    ax.clear()
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(1, -1)  # Invertir eje Y
    ax.set_zlim3d(1, -1)  # Invertir Z para que coincida con MediaPipe
    
    for i, j in ARMS_CONNECTIONS:
        if i < len(keypoints_3d) and j < len(keypoints_3d):
            if keypoints_3d[i][3] > threshold and keypoints_3d[j][3] > threshold:

                x1 = (keypoints_3d[i][0] - 0.5) * 2 
                y1 = (keypoints_3d[i][1] - 0.5) * 2  
                z1 = keypoints_3d[i][2]
                
                x2 = (keypoints_3d[j][0] - 0.5) * 2
                y2 = (keypoints_3d[j][1] - 0.5) * 2  
                z2 = keypoints_3d[j][2]
                
                ax.plot([x1, x2], [z1, z2], [y1, y2], 'b-', linewidth=3)
    
    for idx in ARMS_KEYPOINTS:
        if idx < len(keypoints_3d) and keypoints_3d[idx][3] > threshold:
            x = (keypoints_3d[idx][0] - 0.5) * 2
            y = (keypoints_3d[idx][1] - 0.5) * 2  # Mantener Y normal
            z = keypoints_3d[idx][2]
            ax.scatter([x], [z], [y], c='red', s=100)
    
    ax.set_xlabel('X (horizontal)')
    ax.set_ylabel('Z (profundidad)')
    ax.set_zlabel('Y (vertical)')
    ax.set_title('Vista 3D - YOLO(X,Y) + MediaPipe(Z)')
    

    ax.view_init(elev=10, azim=-60)

def main():

    yolo_model = YOLO("yolo11n-pose.pt")
    pose_processor = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
        smooth_landmarks=True
    )
    
    cap = cv2.VideoCapture(0)
    prev_kpts = None
    
    if not cap.isOpened():
        print("Error al abrir la cámara.")
        return
    
    print("Iniciando detección híbrida YOLO + MediaPipe...")
    print("Presiona 'ESC' para salir")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        img_height, img_width = frame.shape[:2]
        
        results = yolo_model(frame, conf=DETECTION_CONF, imgsz=640, verbose=False)
        r = results[0]
        
        if r.keypoints is None:
            prev_kpts = None
            cv2.imshow("YOLO 2D + MediaPipe Z", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue
        
        kp_tensor = r.keypoints.data
        kp_np = kp_tensor.cpu().numpy()
        
        if kp_np.ndim == 3 and kp_np.shape[0] == 0:
            prev_kpts = None
            cv2.imshow("YOLO 2D + MediaPipe Z", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue
        
        person_kpts = kp_np[0] if kp_np.ndim == 3 else kp_np
        
        if prev_kpts is None:
            smoothed_kpts = person_kpts.copy()
        else:
            smoothed_kpts = (
                SMOOTHING_ALPHA * person_kpts +
                (1 - SMOOTHING_ALPHA) * prev_kpts
            )
        prev_kpts = smoothed_kpts
        
        z_coords, world_coords = get_z_from_mediapipe(frame, smoothed_kpts, pose_processor)
        
        keypoints_3d = create_3d_keypoints(smoothed_kpts, z_coords, world_coords, img_width, img_height, use_hybrid=True)
        
        draw_arms_2d(frame, smoothed_kpts, KP_THRESHOLD)
        
        plot_2d_arms(ax_2d, smoothed_kpts, KP_THRESHOLD, img_width, img_height)
        
        if len(z_coords) > 0: 
            plot_3d_arms(ax_3d, keypoints_3d, KP_THRESHOLD)
        
        plt.pause(0.01)
        
        info_text = f"YOLO keypoints: {len(smoothed_kpts)} | MediaPipe Z coords: {len(z_coords)}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if LEFT_SHOULDER < len(keypoints_3d) and keypoints_3d[LEFT_SHOULDER][3] > KP_THRESHOLD:
            x, y, z = keypoints_3d[LEFT_SHOULDER][:3]
            coord_text = f"L_Shoulder: X:{x:.3f} Y:{y:.3f} Z:{z:.3f}"
            cv2.putText(frame, coord_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        if LEFT_SHOULDER in world_coords:
            mp_coords = world_coords[LEFT_SHOULDER]
            mp_text = f"MP_L_Shoulder: X:{mp_coords['x']:.3f} Y:{mp_coords['y']:.3f} Z:{mp_coords['z']:.3f}"
            cv2.putText(frame, mp_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imshow("YOLO 2D + MediaPipe Z", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()
    plt.close('all')
    pose_processor.close()

if __name__ == "__main__":
    main()