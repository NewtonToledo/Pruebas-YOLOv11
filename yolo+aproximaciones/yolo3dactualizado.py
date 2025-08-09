#!/usr/bin/env python3
import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

def estimate_base_distance_static(k2d, focal_length, avg_shoulder_width):
    if len(k2d) <= max(LEFT_SHOULDER, RIGHT_SHOULDER):
        return 2.0
    if (k2d[LEFT_SHOULDER][2] < KP_THRESHOLD or k2d[RIGHT_SHOULDER][2] < KP_THRESHOLD):
        return 2.0
    lsx, lsy = k2d[LEFT_SHOULDER][:2]; rsx, rsy = k2d[RIGHT_SHOULDER][:2]
    sw_px = math.sqrt((lsx-rsx)**2 + (lsy-rsy)**2)
    if sw_px > 10:
        d = (avg_shoulder_width * float(focal_length)) / sw_px
        return max(0.5, min(d, 5.0))
    return 2.0

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
            ang_r = self.calculate_relative_angle(sh[0],sh[1],el[0],el[1],wr[0],wr[1])
            depth = math.sin(ang_v * DEG2RAD) * self.arm_angle_impact
            ext = ((abs(ang_r)-90)/90 * self.forearm_angle_impact) if abs(ang_r) > 90 else 0
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
    lsx, lsy = k2d[LEFT_SHOULDER][:2]
    rsx, rsy = k2d[RIGHT_SHOULDER][:2]
    cx = ((lsx + rsx) * 0.5) / float(img_w)
    cy = ((lsy + rsy) * 0.5) / float(img_h)
    z_ls = z_coords.get(LEFT_SHOULDER, None)
    z_rs = z_coords.get(RIGHT_SHOULDER, None)
    if z_ls is not None and z_rs is not None:
        cz = 0.5 * (z_ls + z_rs)
    else:
        cz = estimate_base_distance_static(k2d, FOCAL_LENGTH_PX, 0.45)
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
        ok = (i < len(k3d) and j < len(k3d) and k3d[i][3] > KP_THRESHOLD and k3d[j][3] > KP_THRESHOLD)
        if ok and ANCHOR['set']:
            x1 = (k3d[i][0]-ANCHOR['x'])*2.0; y1 = (k3d[i][1]-ANCHOR['y'])*2.0; z1 = (k3d[i][2]-ANCHOR['z'])
            x2 = (k3d[j][0]-ANCHOR['x'])*2.0; y2 = (k3d[j][1]-ANCHOR['y'])*2.0; z2 = (k3d[j][2]-ANCHOR['z'])
            plot3d.lines3d[li].set_data_3d([x1, x2], [z1, z2], [y1, y2])
        else:
            plot3d.lines3d[li].set_data_3d([], [], [])
    for di, idx in enumerate(ARMS_KEYPOINTS):
        if idx < len(k3d) and k3d[idx][3] > KP_THRESHOLD and ANCHOR['set']:
            x = (k3d[idx][0]-ANCHOR['x'])*2.0; y = (k3d[idx][1]-ANCHOR['y'])*2.0; z = (k3d[idx][2]-ANCHOR['z'])
            plot3d.dots3d[di]._offsets3d = ([x], [z], [y])
        else:
            plot3d.dots3d[di]._offsets3d = ([], [], [])

def create_3d_keypoints_geometric(k2d, z_coords, img_w, img_h):
    out=[]
    for i,(x,y,c) in enumerate(k2d):
        if i in ARMS_KEYPOINTS and c > KP_THRESHOLD:
            out.append([x/img_w, y/img_h, z_coords.get(i,0.0), c])
        else:
            out.append([0,0,0,0])
    return np.array(out)

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

        z_coords = z_estimator.estimate_z_hybrid(smoothed_kpts)
        kpts_3d = create_3d_keypoints_geometric(smoothed_kpts, z_coords, img_w, img_h)

        try_set_anchor(smoothed_kpts, z_coords, img_w, img_h)

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
