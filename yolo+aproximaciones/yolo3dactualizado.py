#!/usr/bin/env python3
import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import math
import time
import torch

NUM_THREADS = 1
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(NUM_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_THREADS)

try:
    cv2.setNumThreads(NUM_THREADS)
except Exception:
    pass
try:
    torch.set_num_threads(NUM_THREADS)
    torch.set_num_interop_threads(max(1, NUM_THREADS // 2))
except Exception:
    pass

DETECTION_CONF   = 0.3
KP_THRESHOLD     = 0.2
SMOOTHING_ALPHA  = 0.5

TARGET_WIDTH  = 640
TARGET_HEIGHT = 640
IMGSZ = 448        

PLOT_EVERY_N = 3
SHOW_FPS_OVERLAY = True

FOCAL_LENGTH_PX = 370.0

ANCHOR_CONF = 0.5  

PI = 3.14
DEG2RAD = PI / 180.0
RAD2DEG = 180.0 / PI

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
    def __init__(self):
        self.focal_length = None
        self.avg_shoulder_width = None
        self.arm_angle_impact = None
        self.forearm_angle_impact = None
        self.proportion_weight = None
        self.angle_weight = None

    def distance_2d(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def radian_to_euler(self, radian):
        return radian * RAD2DEG

    def calculate_angle_with_vertical(self, sx, sy, ex, ey):
        v_x = ex - sx; v_y = ey - sy
        return self.radian_to_euler(math.atan2(-v_x, v_y))

    def calculate_relative_angle(self, sx, sy, ex, ey, wx, wy):
        v_x = wx - ex; v_y = wy - ey
        u_x = ex - sx; u_y = ey - sy
        det_v_u = u_x*v_y - u_y*v_x
        dot_v_u = u_x*v_x + u_y*v_y
        return self.radian_to_euler(math.atan2(det_v_u, dot_v_u))

    def estimate_base_distance(self, k2d):
        if len(k2d) <= max(LEFT_SHOULDER, RIGHT_SHOULDER): return 2.0
        if (k2d[LEFT_SHOULDER][2] < KP_THRESHOLD or k2d[RIGHT_SHOULDER][2] < KP_THRESHOLD): return 2.0
        ls = k2d[LEFT_SHOULDER][:2]; rs = k2d[RIGHT_SHOULDER][:2]
        sw_px = self.distance_2d(ls, rs)
        if sw_px > 10:
            d = (self.avg_shoulder_width * float(self.focal_length)) / sw_px
            return max(0.5, min(d, 5.0))
        return 2.0

    def estimate_z_from_proportions(self, k2d):
        base = self.estimate_base_distance(k2d); z = {}
        for idx in ARMS_KEYPOINTS:
            if idx < len(k2d) and k2d[idx][2] > KP_THRESHOLD:
                if idx in [LEFT_ELBOW, RIGHT_ELBOW]:
                    z[idx] = base - 0.1
                elif idx in [LEFT_WRIST, RIGHT_WRIST]:
                    eidx = LEFT_ELBOW if idx == LEFT_WRIST else RIGHT_ELBOW
                    if (eidx < len(k2d) and k2d[eidx][2] > KP_THRESHOLD):
                        ey = k2d[eidx][1]; wy = k2d[idx][1]
                        z[idx] = base + (0.15 if wy > ey else -0.05)
                    else:
                        z[idx] = base
                else:
                    z[idx] = base
        return z

    def analyze_arm_pose(self, k2d, side, base):
        s,e,w = (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST) if side=='left' else (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        zc = {}; idxs=[s,e,w]
        if all(i < len(k2d) and k2d[i][2] > KP_THRESHOLD for i in idxs):
            sh=k2d[s][:2]; el=k2d[e][:2]; wr=k2d[w][:2]
            ang_v = self.calculate_angle_with_vertical(sh[0],sh[1],el[0],el[1])
            depth = math.sin(ang_v * DEG2RAD) * self.arm_angle_impact
            ang_r = self.calculate_relative_angle(sh[0],sh[1],el[0],el[1],wr[0],wr[1])
            abs_r = abs(ang_r)
            ramp = max(0.0, (abs_r - 70.0) / 110.0)   # 0→1 entre 70° y 180°
            ext_sign = 1.0 if ang_r >= 0 else -1.0
            ext = ramp * self.forearm_angle_impact * ext_sign

            # --- modulación por acortamiento (foreshortening) 2D ---
            len_se = self.distance_2d(sh, el) + 1e-6
            len_ew = self.distance_2d(el, wr)
            shortening = 1.0 - min(1.0, len_ew / len_se)   # 0..1 (más acortamiento → más hacia cámara)
            ext *= (0.5 + 0.5 * shortening)
            zc[s]=base; zc[e]=base+depth; zc[w]=base+depth+ext
        return zc

    def estimate_z_from_pose_analysis(self, k2d):
        base = self.estimate_base_distance(k2d)
        z = {}
        z.update(self.analyze_arm_pose(k2d, 'left', base))
        z.update(self.analyze_arm_pose(k2d, 'right', base))
        return z

    def estimate_z_hybrid(self, k2d):
        zp = self.estimate_z_from_proportions(k2d)
        za = self.estimate_z_from_pose_analysis(k2d)
        zf = {}; all_idx = set(zp)|set(za)
        for i in all_idx:
            vp = zp.get(i,0); va = za.get(i,0)
            if vp>0 and va>0: zf[i]=self.proportion_weight*vp + self.angle_weight*va
            elif vp>0: zf[i]=vp
            elif va>0: zf[i]=va
        return zf

ANCHOR = {'set': False, 'x': 0.5, 'y': 0.5, 'z': 0.0}

def try_set_anchor(k2d, z_coords, img_w, img_h):
    if ANCHOR['set']:
        return
    ok_ls = LEFT_SHOULDER < len(k2d) and k2d[LEFT_SHOULDER][2] >= ANCHOR_CONF
    ok_rs = RIGHT_SHOULDER < len(k2d) and k2d[RIGHT_SHOULDER][2] >= ANCHOR_CONF
    if not (ok_ls and ok_rs):
        return

    # Necesitamos Z de ambos hombros; si no están, no anclamos todavía
    z_ls = z_coords.get(LEFT_SHOULDER, None)
    z_rs = z_coords.get(RIGHT_SHOULDER, None)
    if (z_ls is None) or (z_rs is None):
        return

    lsx, lsy = k2d[LEFT_SHOULDER][:2]
    rsx, rsy = k2d[RIGHT_SHOULDER][:2]
    cx = ((lsx + rsx) * 0.5) / float(img_w)
    cy = ((lsy + rsy) * 0.5) / float(img_h)
    cz = 0.5 * (z_ls + z_rs)

    ANCHOR.update({'set': True, 'x': cx, 'y': cy, 'z': float(cz)})


plt.ion()
fig = plt.figure(figsize=(8, 6))
ax_3d = fig.add_subplot(111, projection='3d')
plt.tight_layout()
plt.show(block=False)

class Plot3DState:
    def __init__(self):
        self.lines3d = []
        self.dots3d  = []
        self.ready = False

plot3d = Plot3DState()

def init_plot3d(ax3d):
    ax3d.set_xlim3d(-1, 1)
    ax3d.set_ylim3d( 1,-1)
    ax3d.set_zlim3d( 1,-1)
    ax3d.set_xlabel('X (anclado)'); ax3d.set_ylabel('Z (anclado)'); ax3d.set_zlabel('Y (anclado)')
    ax3d.set_title('Vista 3D - YOLO + Z Geométrica (anclado)')
    ax3d.view_init(elev=10, azim=-60)
    plot3d.lines3d = [ax3d.plot([0,0], [0,0], [0,0], 'b-', linewidth=3)[0] for _ in ARMS_CONNECTIONS]
    plot3d.dots3d  = [ax3d.scatter([0], [0], [0], s=100, c='red') for _ in ARMS_KEYPOINTS]
    plot3d.ready = True

def update_3d_artists(k3d):
    for li, (i, j) in enumerate(ARMS_CONNECTIONS):
        ok = (i < len(k3d) and j < len(k3d) and k3d[i][3] > KP_THRESHOLD and k3d[j][3] > KP_THRESHOLD and ANCHOR['set'])
        if ok:
            x1, y1, z1 = k3d[i][:3]
            x2, y2, z2 = k3d[j][:3]
            plot3d.lines3d[li].set_data_3d([x1, x2], [z1, z2], [y1, y2])  # ojo: Y del plot = tu Z rel
        else:
            plot3d.lines3d[li].set_data_3d([], [], [])
    for di, idx in enumerate(ARMS_KEYPOINTS):
        if idx < len(k3d) and k3d[idx][3] > KP_THRESHOLD and ANCHOR['set']:
            x, y, z = k3d[idx][:3]
            plot3d.dots3d[di]._offsets3d = ([x], [z], [y])
        else:
            plot3d.dots3d[di]._offsets3d = ([], [], [])

def create_3d_keypoints_normalized(k2d, z_coords, img_w, img_h, base_dist, anchor, xy_center=None):
    """
    Devuelve [x_rel, y_rel, z_rel, conf] para cada keypoint de brazos:
      - x_rel, y_rel: (pos_norm - anchor) * 2.0  -> ~[-1,1]
      - z_rel: (z_m - anchor_z) / base_dist     -> adimensional
    """
    out = []
    denom = max(base_dist, 1e-6)
    cx = anchor['x'] if (xy_center is None) else xy_center[0]
    cy = anchor['y'] if (xy_center is None) else xy_center[1]
    for i, (x, y, c) in enumerate(k2d):
        if i in ARMS_KEYPOINTS and c > KP_THRESHOLD and (i in z_coords):
            xn = x / float(img_w)
            yn = y / float(img_h)
            x_rel = (xn - cx) * 2.0
            y_rel = (yn - cy) * 2.0
            z_rel = (z_coords[i] - anchor['z']) / denom
            out.append([x_rel, y_rel, z_rel, c])
        else:
            out.append([0, 0, 0, 0])
    return np.array(out)

def get_xy_center_current(k2d, img_w, img_h, fallback_anchor):
    """Devuelve (cx, cy) normalizados del centro de hombros del FRAME ACTUAL.
       Si no hay hombros confiables, usa el anchor como respaldo."""
    ok_ls = LEFT_SHOULDER < len(k2d) and k2d[LEFT_SHOULDER][2] > KP_THRESHOLD
    ok_rs = RIGHT_SHOULDER < len(k2d) and k2d[RIGHT_SHOULDER][2] > KP_THRESHOLD
    if ok_ls and ok_rs:
        lsx, lsy = k2d[LEFT_SHOULDER][:2]
        rsx, rsy = k2d[RIGHT_SHOULDER][:2]
        cx = ((lsx + rsx) * 0.5) / float(img_w)
        cy = ((lsy + rsy) * 0.5) / float(img_h)
        return cx, cy
    return fallback_anchor['x'], fallback_anchor['y']


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)

    ret_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ret_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Cam real: {ret_w}x{ret_h}")

    yolo_model = YOLO("modelos/yolo11n-pose.pt")
    z_estimator = GeometricZEstimator()
    z_estimator.focal_length = FOCAL_LENGTH_PX
    z_estimator.avg_shoulder_width = 0.45
    z_estimator.arm_angle_impact = 0.2
    z_estimator.forearm_angle_impact = 0.15
    z_estimator.proportion_weight = 0.25
    z_estimator.angle_weight = 0.75

    if not cap.isOpened():
        print("Error al abrir la camara."); return

    print("Iniciando deteccion YOLO + Estimacion Geometrica Z...")
    print("Presiona 'ESC' para salir")

    prev_kpts = None
    t0 = time.perf_counter(); last_fps = 0.0
    frame_idx = 0

    if not plot3d.ready:
        init_plot3d(ax_3d)

    while True:
        ret, frame = cap.read()
        if not ret: break
        img_h, img_w = frame.shape[:2]

        with torch.inference_mode():
            results = yolo_model(
                frame, conf=DETECTION_CONF, imgsz=IMGSZ,
                verbose=False, max_det=1
            )
        r = results[0]

        if r.keypoints is None:
            prev_kpts = None
            cv2.imshow("YOLO + Geometric Z Estimation", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
            continue

        kp_tensor = r.keypoints.data
        kp_np = kp_tensor.cpu().numpy()
        if kp_np.ndim == 3 and kp_np.shape[0] == 0:
            prev_kpts = None
            cv2.imshow("YOLO + Geometric Z Estimation", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
            continue

        person_kpts = kp_np[0] if kp_np.ndim == 3 else kp_np

        smoothed_kpts = person_kpts.copy() if prev_kpts is None else (
            SMOOTHING_ALPHA*person_kpts + (1-SMOOTHING_ALPHA)*prev_kpts
        )
        prev_kpts = smoothed_kpts

        # 1) Estimación Z (m)
        z_coords = z_estimator.estimate_z_hybrid(smoothed_kpts)

        # 2) Distancia base del frame (m) — la usaremos para normalizar Z
        base_dist = z_estimator.estimate_base_distance(smoothed_kpts)

        # 3) Intentar fijar ancla (usa hombros + z ya estimada)
        try_set_anchor(smoothed_kpts, z_coords, img_w, img_h)
        # Centro de hombros del frame actual para centrar X,Y (dinámico)
        xy_center = get_xy_center_current(smoothed_kpts, img_w, img_h, ANCHOR)


        # 4) Construir kpts 3D en UNIDADES NORMALIZADAS (relativas al ancla)
        if ANCHOR['set']:
            kpts_3d = create_3d_keypoints_normalized(
                smoothed_kpts, z_coords, img_w, img_h, base_dist, ANCHOR, xy_center=xy_center
            )
        else:
            # sin ancla aún → no dibujar (conf=0)
            kpts_3d = np.zeros((len(smoothed_kpts), 4), dtype=float)


        #Overlays informativos
        info_text = f"YOLO keypoints: {len(smoothed_kpts)} | Z coords: {len(z_coords)}"
        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if LEFT_SHOULDER < len(kpts_3d) and kpts_3d[LEFT_SHOULDER][3] > KP_THRESHOLD:
            x, y, z = kpts_3d[LEFT_SHOULDER][:3]
            cv2.putText(frame, f"L_Shoulder: X:{x:.3f} Y:{y:.3f} Z:{z:.3f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        base_dist = z_estimator.estimate_base_distance(smoothed_kpts)
        anch = "OK" if ANCHOR['set'] else "…"
        cv2.putText(frame,
                    f"Dist est.: {base_dist:.2f}m | f(px): {z_estimator.focal_length:.1f} | Anchor: {anch}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Plot 3D: actualizar artistas cada N frames
        frame_idx += 1
        if frame_idx % PLOT_EVERY_N == 0 and ANCHOR['set'] and len(z_coords) > 0:
            update_3d_artists(kpts_3d)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.0001)

        # FPS (aprox) cada 10 frames
        if SHOW_FPS_OVERLAY:
            if frame_idx % 10 == 0:
                t1 = time.perf_counter()
                last_fps = 10.0 / (t1 - t0) if (t1 - t0) > 0 else 0.0
                t0 = t1
            if last_fps > 0:
                cv2.putText(frame, f"FPS aprox: {last_fps:.1f}",
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,255,50), 2)

        cv2.imshow("YOLO + Geometric Z Estimation", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()
    plt.close('all')

if __name__ == "__main__":
    main()
