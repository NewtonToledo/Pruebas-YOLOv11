#!/usr/bin/env python3
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

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

class GeometricZEstimator:
    def __init__(self, focal_length=500, avg_shoulder_width=0.40, 
                 arm_angle_impact=0.2, forearm_angle_impact=0.15,
                 proportion_weight=0.6, angle_weight=0.4):
        self.focal_length = focal_length
        self.avg_shoulder_width = avg_shoulder_width
        
        self.arm_angle_impact = arm_angle_impact        
        self.forearm_angle_impact = forearm_angle_impact 
        self.proportion_weight = proportion_weight       # Peso del metodo de proporciones
        self.angle_weight = angle_weight                 # Peso del metodo angular
        
        self.body_ratios = {
            'upper_arm': 0.188,
            'forearm': 0.146,
            'torso': 0.288,
            'thigh': 0.245,
            'shin': 0.246
        }
    
    def distance_2d(self, p1, p2):
        """Calcula distancia euclidiana entre dos puntos 2D"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def radian_to_euler(self, radian):
        """Convierte radianes a grados"""
        return radian * 180.0 / math.pi
    
    def calculate_angle_with_vertical(self, shoulder_x, shoulder_y, elbow_x, elbow_y):
        """Calcula ángulo del brazo superior con respecto a la vertical"""
        v_x = elbow_x - shoulder_x
        v_y = elbow_y - shoulder_y
        return self.radian_to_euler(math.atan2(-v_x, v_y))
    
    def calculate_angle_with_vertical_zy(self, shoulder_z, shoulder_y, elbow_z, elbow_y):
        """Calcula ángulo con respecto a la vertical en plano ZY"""
        v_z = elbow_z - shoulder_z
        v_y = elbow_y - shoulder_y
        return self.radian_to_euler(math.atan2(-v_z, v_y))
    
    def calculate_relative_angle(self, shoulder_x, shoulder_y, elbow_x, elbow_y, wrist_x, wrist_y):
        """Calcula ángulo relativo entre brazo superior y antebrazo"""
        v_x = wrist_x - elbow_x
        v_y = wrist_y - elbow_y
        u_x = elbow_x - shoulder_x
        u_y = elbow_y - shoulder_y
        det_v_u = u_x * v_y - u_y * v_x
        dot_v_u = u_x * v_x + u_y * v_y
        return self.radian_to_euler(math.atan2(det_v_u, dot_v_u))
    
    def calculate_relative_angle_zy(self, shoulder_z, shoulder_y, elbow_z, elbow_y, wrist_z, wrist_y):
        """Calcula ángulo relativo en plano ZY"""
        v_z = wrist_z - elbow_z
        v_y = wrist_y - elbow_y
        u_z = elbow_z - shoulder_z
        u_y = elbow_y - shoulder_y
        det_v_u = u_z * v_y - u_y * v_z
        dot_v_u = u_z * v_z + u_y * v_y
        return self.radian_to_euler(math.atan2(det_v_u, dot_v_u))
    
    def estimate_base_distance(self, keypoints_2d):
        """Estima distancia base usando ancho de hombros"""
        if len(keypoints_2d) <= max(LEFT_SHOULDER, RIGHT_SHOULDER):
            return 2.0
            
        left_shoulder = keypoints_2d[LEFT_SHOULDER][:2]
        right_shoulder = keypoints_2d[RIGHT_SHOULDER][:2]
        
        # Verificar confianza de los hombros
        if (keypoints_2d[LEFT_SHOULDER][2] < KP_THRESHOLD or 
            keypoints_2d[RIGHT_SHOULDER][2] < KP_THRESHOLD):
            return 2.0
        
        shoulder_width_px = self.distance_2d(left_shoulder, right_shoulder)
        
        if shoulder_width_px > 10: 
            # Formula de perspectiva
            base_distance = (self.avg_shoulder_width * self.focal_length) / shoulder_width_px
            return max(0.5, min(base_distance, 5.0))  # Limitar entre 0.5 y 5 metros
        return 2.0 
    
    def estimate_z_from_proportions(self, keypoints_2d):
        """Estima Z usando proporciones corporales"""
        base_distance = self.estimate_base_distance(keypoints_2d)
        z_coords = {}
        
        for idx in ARMS_KEYPOINTS:
            if idx < len(keypoints_2d) and keypoints_2d[idx][2] > KP_THRESHOLD:
                if idx in [LEFT_ELBOW, RIGHT_ELBOW]: 
                    z_coords[idx] = base_distance - 0.1
                elif idx in [LEFT_WRIST, RIGHT_WRIST]: 
                    elbow_idx = LEFT_ELBOW if idx == LEFT_WRIST else RIGHT_ELBOW
                    if (elbow_idx < len(keypoints_2d) and 
                        keypoints_2d[elbow_idx][2] > KP_THRESHOLD):
                        elbow_y = keypoints_2d[elbow_idx][1]
                        wrist_y = keypoints_2d[idx][1]
                        z_offset = 0.15 if wrist_y > elbow_y else -0.05
                        z_coords[idx] = base_distance + z_offset
                    else:
                        z_coords[idx] = base_distance
                else: 
                    z_coords[idx] = base_distance
        
        return z_coords
    
    def analyze_arm_pose(self, keypoints_2d, side, base_distance):
        """Analiza pose de un brazo específico usando cálculos avanzados de ángulos"""
        if side == 'left':
            shoulder_idx, elbow_idx, wrist_idx = LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
        else:
            shoulder_idx, elbow_idx, wrist_idx = RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
        
        z_coords = {}
        
        # Verificar que todos los puntos existan
        indices = [shoulder_idx, elbow_idx, wrist_idx]
        if all(idx < len(keypoints_2d) and keypoints_2d[idx][2] > KP_THRESHOLD for idx in indices):
            
            shoulder = keypoints_2d[shoulder_idx][:2]
            elbow = keypoints_2d[elbow_idx][:2]
            wrist = keypoints_2d[wrist_idx][:2]
            
            # Calcular ángulos usando las funciones avanzadas
            arm_angle_vertical = self.calculate_angle_with_vertical(
                shoulder[0], shoulder[1], elbow[0], elbow[1]
            )
            
            relative_angle = self.calculate_relative_angle(
                shoulder[0], shoulder[1], elbow[0], elbow[1], wrist[0], wrist[1]
            )
            
            # Convertir angulos a factores de profundidad Z
            # Angulos cercanos a 0° (vertical) = menos variación Z
            # Angulos cercanos a +-90° (horizontal) = mas variación Z
            
            # Normalizar ángulo vertical a factor (-1 a 1)
            arm_depth_factor = math.sin(math.radians(arm_angle_vertical)) * self.arm_angle_impact
            
            # Normalizar ángulo relativo a factor de extension
            # Angulos > 90° = brazo extendido hacia adelante
            # Angulos < 90° = brazo flexionado
            extension_factor = 0
            if abs(relative_angle) > 90:
                extension_factor = (abs(relative_angle) - 90) / 90 * self.forearm_angle_impact
            
            # Asignar coordenadas Z 
            z_coords[shoulder_idx] = base_distance
            z_coords[elbow_idx] = base_distance + arm_depth_factor
            z_coords[wrist_idx] = base_distance + arm_depth_factor + extension_factor
        
        return z_coords
    
    def estimate_z_from_pose_analysis(self, keypoints_2d):
        """Analiza la pose para inferir orientaciones 3D"""
        z_coords = {}
        base_distance = self.estimate_base_distance(keypoints_2d)
        
        left_arm_z = self.analyze_arm_pose(keypoints_2d, 'left', base_distance)
        right_arm_z = self.analyze_arm_pose(keypoints_2d, 'right', base_distance)
        
        z_coords.update(left_arm_z)
        z_coords.update(right_arm_z)
        
        return z_coords
    
    def estimate_z_hybrid(self, keypoints_2d):
        # Metodo 1: Proporciones
        z_props = self.estimate_z_from_proportions(keypoints_2d)
        
        # Metodo 2: Analisis de pose
        z_pose = self.estimate_z_from_pose_analysis(keypoints_2d)
        
        # Combinar resultados con pesos
        z_final = {}
        all_indices = set(z_props.keys()) | set(z_pose.keys())
        
        for idx in all_indices:
            z_prop_val = z_props.get(idx, 0)
            z_pose_val = z_pose.get(idx, 0)
            
            if z_prop_val > 0 and z_pose_val > 0:
                # Promedio ponderado usando pesos configurables
                z_final[idx] = self.proportion_weight * z_prop_val + self.angle_weight * z_pose_val
            elif z_prop_val > 0:
                z_final[idx] = z_prop_val
            elif z_pose_val > 0:
                z_final[idx] = z_pose_val
        
        return z_final

# Configurar matplotlib
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

def create_3d_keypoints_geometric(yolo_keypoints, z_coords, img_width, img_height):
    """Crea keypoints 3D usando estimación geométrica de Z"""
    keypoints_3d = []    
    for i, (x, y, conf) in enumerate(yolo_keypoints):
        if i in ARMS_KEYPOINTS and conf > KP_THRESHOLD:
            x_norm = x / img_width
            y_norm = y / img_height
            z = z_coords.get(i, 0.0)
            keypoints_3d.append([x_norm, y_norm, z, conf])
        else:
            keypoints_3d.append([0, 0, 0, 0]) 
    return np.array(keypoints_3d)

def draw_arms_2d(img, keypoints, threshold):
    """Dibuja brazos en la imagen 2D"""
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
    """Grafica brazos en 2D normalizado"""
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
    """Grafica brazos en 3D"""
    ax.clear()
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(1, -1)  # Invertir eje Y
    ax.set_zlim3d(1, -1)  # Invertir Z
    
    for i, j in ARMS_CONNECTIONS:
        if i < len(keypoints_3d) and j < len(keypoints_3d):
            if keypoints_3d[i][3] > threshold and keypoints_3d[j][3] > threshold:
                # Transformar coordenadas para visualización 3D
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
            y = (keypoints_3d[idx][1] - 0.5) * 2
            z = keypoints_3d[idx][2]
            ax.scatter([x], [z], [y], c='red', s=100)
    
    ax.set_xlabel('X (horizontal)')
    ax.set_ylabel('Z (profundidad)')
    ax.set_zlabel('Y (vertical)')
    ax.set_title('Vista 3D - YOLO + Estimacion Geometrica Z')
    ax.view_init(elev=10, azim=-60)

def main():
    # Inicializar YOLO y estimador 
    yolo_model = YOLO("yolo11n-pose.pt")
    z_estimator = GeometricZEstimator(
        focal_length=370,             
        avg_shoulder_width=0.45,       
        arm_angle_impact=0.2,         
        forearm_angle_impact=0.15,     
        proportion_weight=0.4,         
        angle_weight=0.6               
    )
    
    cap = cv2.VideoCapture(0)
    prev_kpts = None
    
    if not cap.isOpened():
        print("Error al abrir la camara.")
        return
    
    print("Iniciando deteccion YOLO + Estimacion Geometrica Z...")
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
            cv2.imshow("YOLO + Geometric Z Estimation", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue
        
        kp_tensor = r.keypoints.data
        kp_np = kp_tensor.cpu().numpy()
        
        if kp_np.ndim == 3 and kp_np.shape[0] == 0:
            prev_kpts = None
            cv2.imshow("YOLO + Geometric Z Estimation", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue
        
        person_kpts = kp_np[0] if kp_np.ndim == 3 else kp_np
        
        # Suavizado temporal
        if prev_kpts is None:
            smoothed_kpts = person_kpts.copy()
        else:
            smoothed_kpts = (
                SMOOTHING_ALPHA * person_kpts +
                (1 - SMOOTHING_ALPHA) * prev_kpts
            )
        prev_kpts = smoothed_kpts
        
        z_coords = z_estimator.estimate_z_hybrid(smoothed_kpts)
        
        keypoints_3d = create_3d_keypoints_geometric(smoothed_kpts, z_coords, img_width, img_height)
        
        draw_arms_2d(frame, smoothed_kpts, KP_THRESHOLD)
        
        plot_2d_arms(ax_2d, smoothed_kpts, KP_THRESHOLD, img_width, img_height)
        
        if len(z_coords) > 0:
            plot_3d_arms(ax_3d, keypoints_3d, KP_THRESHOLD)
        
        plt.pause(0.01)
        
        info_text = f"YOLO keypoints: {len(smoothed_kpts)} | Z coords estimadas: {len(z_coords)}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if LEFT_SHOULDER < len(keypoints_3d) and keypoints_3d[LEFT_SHOULDER][3] > KP_THRESHOLD:
            x, y, z = keypoints_3d[LEFT_SHOULDER][:3]
            coord_text = f"L_Shoulder: X:{x:.3f} Y:{y:.3f} Z:{z:.3f}"
            cv2.putText(frame, coord_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        base_dist = z_estimator.estimate_base_distance(smoothed_kpts)
        dist_text = f"Distancia estimada: {base_dist:.2f}m"
        cv2.putText(frame, dist_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        if LEFT_SHOULDER < len(smoothed_kpts) and LEFT_ELBOW < len(smoothed_kpts):
            if (smoothed_kpts[LEFT_SHOULDER][2] > KP_THRESHOLD and 
                smoothed_kpts[LEFT_ELBOW][2] > KP_THRESHOLD):
                
                shoulder = smoothed_kpts[LEFT_SHOULDER][:2]
                elbow = smoothed_kpts[LEFT_ELBOW][:2]
                
                arm_angle = z_estimator.calculate_angle_with_vertical(
                    shoulder[0], shoulder[1], elbow[0], elbow[1]
                )
                angle_text = f"Angulo brazo izq: {arm_angle:.1f}°"
                cv2.putText(frame, angle_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        cv2.imshow("YOLO + Geometric Z Estimation", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()
    plt.close('all')

if __name__ == "__main__":
    main()