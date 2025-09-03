#!/usr/bin/env python3
import os
import time
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

# ===================== Configuración de performance =====================
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

# ===================== Parámetros globales =====================
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

PI = 3.141592653589793
DEG2RAD = PI / 180.0
RAD2DEG = 180.0 / PI

# ===================== Índices de keypoints (Ultralytics pose) =====================
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

# ===================== ÁNGULOS ARTICULARES (reales) =====================
def _norm(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v, n

def _angle_between(u, v):
    """Ángulo no firmado entre u y v en [0, 180] grados."""
    u, nu = _norm(u); v, nv = _norm(v)
    if nu < 1e-6 or nv < 1e-6:
        return None
    cosang = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return math.degrees(math.acos(cosang))

def _signed_angle_in_plane(a, b, n):
    """
    Ángulo firmado (b -> a) en el plano perpendicular a n.
    Devuelve grados en [-180, 180].
    """
    a = np.asarray(a, float); b = np.asarray(b, float); n = np.asarray(n, float)
    n, nn = _norm(n)
    if nn < 1e-6:
        return None
    a_proj = a - np.dot(a, n) * n
    b_proj = b - np.dot(b, n) * n
    _, na = _norm(a_proj); _, nb = _norm(b_proj)
    if na < 1e-6 or nb < 1e-6:
        return None
    cross = np.cross(b_proj, a_proj)
    sin_ = float(np.dot(n, cross))
    cos_ = float(np.dot(b_proj, a_proj))
    ang = math.degrees(math.atan2(sin_, cos_))
    return ang  # [-180, 180]

def compute_arm_angles(kpts_3d, side, kp_thresh=0.2):
    """
    kpts_3d: np.array(N,4) con [x, y, z, conf] (se construye abajo).
    side: 'left' o 'right'
    Devuelve dict con:
      - elbow_flex            (0..180 aprox; 0 ≈ extendido)
      - shoulder_abduction    (+: brazo hacia lateral en plano frontal)
      - shoulder_flexion      (+: brazo al frente en plano sagital)
    Sin clamps/saturaciones.
    """
    s = LEFT_SHOULDER if side == 'left' else RIGHT_SHOULDER
    e = LEFT_ELBOW    if side == 'left' else RIGHT_ELBOW
    w = LEFT_WRIST    if side == 'left' else RIGHT_WRIST

    if (s >= len(kpts_3d)) or (e >= len(kpts_3d)) or (w >= len(kpts_3d)):
        return None
    if (kpts_3d[s,3] < kp_thresh) or (kpts_3d[e,3] < kp_thresh) or (kpts_3d[w,3] < kp_thresh):
        return None

    S = kpts_3d[s,:3]; E = kpts_3d[e,:3]; W = kpts_3d[w,:3]
    upper   = E - S  # brazo superior
    forearm = W - E  # antebrazo

    elbow = _angle_between(upper, forearm)
    if elbow is None:
        return None

    vertical  = np.array([0.0, -1.0, 0.0])  # -Y ~ arriba (signo no crítico para magnitud)
    n_frontal = np.array([0.0, 0.0, 1.0])   # normal al plano frontal (XOY)
    n_sagital = np.array([1.0, 0.0, 0.0])   # normal al plano sagital (YOZ)

    abd  = _signed_angle_in_plane(upper, vertical, n_frontal)  # +: lateral
    flex = _signed_angle_in_plane(upper, vertical, n_sagital)  # +: frente
    if abd is None or flex is None:
        return None

    return {
        'elbow_flex'         : elbow,
        'shoulder_abduction' : abd,
        'shoulder_flexion'   : flex,
    }

# ===================== Estimador Z (solo hombros/proporciones) =====================
class GeometricZEstimator:
    """
    Simplificado: sin impactos por ángulos.
    Usa anchura de hombros + focal para una distancia base y
    pequeñas reglas de proporción para codo/muñeca.
    """
    def __init__(self):
        self.focal_length = None
        self.avg_shoulder_width = None
        self.proportion_weight = 1.0

    @staticmethod
    def distance_2d(p1, p2):
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    def estimate_base_distance(self, k2d):
        if len(k2d) <= max(LEFT_SHOULDER, RIGHT_SHOULDER):
            return 2.0
        if (k2d[LEFT_SHOULDER][2] < KP_THRESHOLD or k2d[RIGHT_SHOULDER][2] < KP_THRESHOLD):
            return 2.0
        ls = k2d[LEFT_SHOULDER][:2]; rs = k2d[RIGHT_SHOULDER][:2]
        sw_px = self.distance_2d(ls, rs)
        if sw_px > 10:
            d = (self.avg_shoulder_width * float(self.focal_length)) / sw_px
            return max(0.5, min(d, 5.0))
        return 2.0

    def estimate_z_from_proportions(self, k2d):
        base = self.estimate_base_distance(k2d)
        z = {}
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

    def estimate_z_hybrid(self, k2d):
        zp = self.estimate_z_from_proportions(k2d)
        zf = {}
        for i, val in zp.items():
            zf[i] = self.proportion_weight * val
        return zf


# ===================== Anchor y normalización a 3D relativo =====================
ANCHOR = {'set': False, 'x': 0.5, 'y': 0.5, 'z': 0.0}

def try_set_anchor(k2d, z_coords, img_w, img_h):
    if ANCHOR['set']:
        return
    ok_ls = LEFT_SHOULDER < len(k2d) and k2d[LEFT_SHOULDER][2] >= ANCHOR_CONF
    ok_rs = RIGHT_SHOULDER < len(k2d) and k2d[RIGHT_SHOULDER][2] >= ANCHOR_CONF
    if not (ok_ls and ok_rs):
        return
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

def create_3d_keypoints_normalized(k2d, z_coords, img_w, img_h, base_dist, anchor, xy_center=None):
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
    ok_ls = LEFT_SHOULDER < len(k2d) and k2d[LEFT_SHOULDER][2] > KP_THRESHOLD
    ok_rs = RIGHT_SHOULDER < len(k2d) and k2d[RIGHT_SHOULDER][2] > KP_THRESHOLD
    if ok_ls and ok_rs:
        lsx, lsy = k2d[LEFT_SHOULDER][:2]
        rsx, rsy = k2d[RIGHT_SHOULDER][:2]
        cx = ((lsx + rsx) * 0.5) / float(img_w)
        cy = ((lsy + rsy) * 0.5) / float(img_h)
        return cx, cy
    return fallback_anchor['x'], fallback_anchor['y']

# ===================== Plot 3D (ligero) =====================
plt.ion()
fig = plt.figure(figsize=(6, 4), dpi=72)
ax_3d = fig.add_subplot(111, projection='3d')
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
    ax3d.set_xlabel('X'); ax3d.set_ylabel('Z'); ax3d.set_zlabel('Y')
    ax3d.set_title('3D Pose (fast)')
    ax3d.view_init(elev=10, azim=-60)
    ax3d.grid(False)
    ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])
    try:
        ax3d.set_box_aspect((1,1,1))
    except Exception:
        pass
    plot3d.lines3d = [ax3d.plot([0,0], [0,0], [0,0], '-', linewidth=2)[0] for _ in ARMS_CONNECTIONS]
    plot3d.dots3d  = ax3d.scatter([], [], [], s=40, depthshade=False)
    plot3d.ready = True

def update_3d_artists(k3d):
    xs, ys, zs = [], [], []
    for li, (i, j) in enumerate(ARMS_CONNECTIONS):
        ok = (i < len(k3d) and j < len(k3d) and
              k3d[i][3] > KP_THRESHOLD and k3d[j][3] > KP_THRESHOLD and ANCHOR['set'])
        if ok:
            x1, y1, z1 = k3d[i][:3]
            x2, y2, z2 = k3d[j][:3]
            plot3d.lines3d[li].set_data_3d([x1, x2], [z1, z2], [y1, y2])  # (X, Z, Y)
            xs.extend([x1, x2]); ys.extend([y1, y2]); zs.extend([z1, z2])
        else:
            plot3d.lines3d[li].set_data_3d([], [], [])
    if xs:
        plot3d.dots3d._offsets3d = (xs, zs, ys)
    else:
        plot3d.dots3d._offsets3d = ([], [], [])

# ===================== Main =====================
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
    z_estimator.proportion_weight = 1.0

    if not cap.isOpened():
        print("Error al abrir la camara.")
        return

    print("Iniciando detección YOLO + Estimación Z (proporcional) + Cálculo de ángulos.")
    print("Presiona 'ESC' para salir")

    prev_kpts = None
    t0 = time.perf_counter(); last_fps = 0.0
    frame_idx = 0

    if not plot3d.ready:
        init_plot3d(ax_3d)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_h, img_w = frame.shape[:2]

        with torch.inference_mode():
            results = yolo_model(frame, conf=DETECTION_CONF, imgsz=IMGSZ, verbose=False, max_det=1)
        r = results[0]

        if r.keypoints is None:
            prev_kpts = None
            cv2.imshow("YOLO + 3D Arms", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
            continue

        kp_tensor = r.keypoints.data
        kp_np = kp_tensor.cpu().numpy()
        if kp_np.ndim == 3 and kp_np.shape[0] == 0:
            prev_kpts = None
            cv2.imshow("YOLO + 3D Arms", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
            continue

        person_kpts = kp_np[0] if kp_np.ndim == 3 else kp_np

        # Suavizado temporal simple
        smoothed_kpts = person_kpts.copy() if prev_kpts is None else (
            SMOOTHING_ALPHA*person_kpts + (1-SMOOTHING_ALPHA)*prev_kpts
        )
        prev_kpts = smoothed_kpts

        # Z por proporciones (sin ángulos)
        z_coords  = z_estimator.estimate_z_hybrid(smoothed_kpts)
        base_dist = z_estimator.estimate_base_distance(smoothed_kpts)

        # Anchor y centro XY
        try_set_anchor(smoothed_kpts, z_coords, img_w, img_h)
        xy_center = get_xy_center_current(smoothed_kpts, img_w, img_h, ANCHOR)

        # Construcción de kpts_3d
        if ANCHOR['set']:
            kpts_3d = create_3d_keypoints_normalized(
                smoothed_kpts, z_coords, img_w, img_h, base_dist, ANCHOR, xy_center=xy_center
            )
        else:
            kpts_3d = np.zeros((len(smoothed_kpts), 4), dtype=float)

        # --- Ángulos articulares (solo overlay, sin clamps ni mapeos) ---
        if ANCHOR['set'] and len(kpts_3d) > 0:
            L = compute_arm_angles(kpts_3d, 'left',  kp_thresh=KP_THRESHOLD)
            R = compute_arm_angles(kpts_3d, 'right', kp_thresh=KP_THRESHOLD)

            if L:
                lf = L['elbow_flex']; la = L['shoulder_abduction']; lx = L['shoulder_flexion']
                cv2.putText(frame, f"L-elbow:{(lf or 0):.0f}  L-abd:{(la or 0):.0f}  L-flex:{(lx or 0):.0f}", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)
            if R:
                rf = R['elbow_flex']; ra = R['shoulder_abduction']; rx = R['shoulder_flexion']
                cv2.putText(frame, f"R-elbow:{(rf or 0):.0f}  R-abd:{(ra or 0):.0f}  R-flex:{(rx or 0):.0f}", (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)

        # Overlays informativos base
        info_text = f"YOLO keypoints: {len(smoothed_kpts)} | Z coords: {len(z_coords)}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if LEFT_SHOULDER < len(kpts_3d) and kpts_3d[LEFT_SHOULDER][3] > KP_THRESHOLD:
            x, y, z = kpts_3d[LEFT_SHOULDER][:3]
            cv2.putText(frame, f"L_Shoulder: X:{x:.3f} Y:{y:.3f} Z:{z:.3f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        anch = "OK" if ANCHOR['set'] else "…"
        cv2.putText(frame,
                    f"Dist est.: {base_dist:.2f}m | f(px): {FOCAL_LENGTH_PX:.1f} | Anchor: {anch}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Plot 3D cada N frames
        frame_idx += 1
        if frame_idx % PLOT_EVERY_N == 0 and ANCHOR['set'] and len(z_coords) > 0:
            update_3d_artists(kpts_3d)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.0001)

        # FPS aprox
        if SHOW_FPS_OVERLAY:
            if frame_idx % 10 == 0:
                t1 = time.perf_counter()
                last_fps = 10.0 / (t1 - t0) if (t1 - t0) > 0 else 0.0
                t0 = t1
            if last_fps > 0:
                cv2.putText(frame, f"FPS aprox: {last_fps:.1f}",
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,255,50), 2)

        cv2.imshow("YOLO + 3D Arms", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.close('all')

if __name__ == "__main__":
    main()
