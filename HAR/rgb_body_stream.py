"""
Stream de cámara RGB con YOLO-pose.

YOLO-pose (yolov8n-pose.pt) detecta personas y estima 17 keypoints COCO en
una sola pasada PyTorch, sin TensorFlow. El stream convierte esos keypoints
al formato BODY_18 con pseudo-profundidad Z=0.

Frame dict que produce frames():
    {
        "image_bgr":  np.ndarray (H,W,3) BGR anotado con YOLO,
        "bodies": [{
            "kp3d":           (18,3) pseudo-3D metros (Z=0),
            "kp_conf":        (18,)  confianza [0,100],
            "id":             int  tracking ID,
            "confidence":     float detección [0,100],
            "tracking_state": "ok",
        }, ...],
        "yolo_ms": float  tiempo de inferencia YOLO en ms,
    }
"""

import time
import cv2
import numpy as np
from ultralytics import YOLO

_YOLO_POSE_MODEL = "yolov8n-pose.pt"
_CONF_THRESH     = 0.2   # confianza mínima por joint

# ── Mapeo COCO-17 → BODY_18 ───────────────────────────────────────────────────
# COCO-17: nose=0, l_eye=1, r_eye=2, l_ear=3, r_ear=4,
#           l_shoulder=5, r_shoulder=6, l_elbow=7, r_elbow=8,
#           l_wrist=9, r_wrist=10, l_hip=11, r_hip=12,
#           l_knee=13, r_knee=14, l_ankle=15, r_ankle=16
#
# BODY_18: 0=nose, 1=neck*, 2=r_shoulder, 3=r_elbow, 4=r_wrist,
#           5=l_shoulder, 6=l_elbow, 7=l_wrist, 8=r_hip, 9=r_knee,
#           10=r_ankle, 11=l_hip, 12=l_knee, 13=l_ankle,
#           14=r_eye, 15=l_eye, 16=r_ear, 17=l_ear
# (*) neck no existe en COCO → se estima como promedio de hombros.
_B18_FROM_COCO = [
    0,    # 0  nose
    None, # 1  neck  ← sintético
    6,    # 2  r_shoulder
    8,    # 3  r_elbow
    10,   # 4  r_wrist
    5,    # 5  l_shoulder
    7,    # 6  l_elbow
    9,    # 7  l_wrist
    12,   # 8  r_hip
    14,   # 9  r_knee
    16,   # 10 r_ankle
    11,   # 11 l_hip
    13,   # 12 l_knee
    15,   # 13 l_ankle
    2,    # 14 r_eye
    1,    # 15 l_eye
    4,    # 16 r_ear
    3,    # 17 l_ear
]


def _yolo_to_body18_2d(kp_yolo: np.ndarray) -> tuple:
    """
    Convierte keypoints YOLO-pose a BODY_18 en píxeles.

    YOLO-pose da [x_pixel, y_pixel, conf] en el frame completo.
    Agrega cuello sintético como promedio de hombros.

    Returns: kp2d (18,2), kp_conf (18,)
    """
    kp2d    = np.full((18, 2), np.nan, dtype=np.float32)
    kp_conf = np.zeros(18, dtype=np.float32)

    for b18, coco in enumerate(_B18_FROM_COCO):
        if coco is None:
            continue
        x, y, c = kp_yolo[coco]
        kp_conf[b18] = float(c) * 100.0
        if float(c) >= _CONF_THRESH:
            kp2d[b18] = [float(x), float(y)]

    # Cuello sintético = promedio de hombros disponibles
    r_sh, l_sh = kp2d[2], kp2d[5]
    rc, lc     = kp_conf[2], kp_conf[5]
    if np.isfinite(r_sh).all() and np.isfinite(l_sh).all():
        kp2d[1], kp_conf[1] = 0.5 * (r_sh + l_sh), 0.5 * (rc + lc)
    elif np.isfinite(r_sh).all():
        kp2d[1], kp_conf[1] = r_sh.copy(), rc
    elif np.isfinite(l_sh).all():
        kp2d[1], kp_conf[1] = l_sh.copy(), lc

    return kp2d, kp_conf


def _to_pseudo3d(kp2d: np.ndarray, bbox_h: float, ref_h: float):
    """
    Eleva BODY_18 2D (píxeles) a pseudo-3D métrico con Z=0.

    Escala = bbox_height_px / ref_height_m  (píxeles por metro).
    Origen en pelvis (promedio de caderas). Y invertido (arriba = positivo).

    Returns:
        kp3d      (18,3) float32 — coordenadas métricas, origen en pelvis
        pelvis_px (2,)   float32 — posición de la pelvis en píxeles del frame
        ppm       float          — píxeles por metro (para back-proyección)
    """
    ppm = max(bbox_h, 1.0) / ref_h

    hips      = [kp2d[i] for i in [8, 11] if np.isfinite(kp2d[i]).all()]
    pelvis_px = np.mean(hips, axis=0).astype(np.float32) if hips else (
        kp2d[np.isfinite(kp2d).all(axis=1)].mean(axis=0).astype(np.float32)
        if np.isfinite(kp2d).any() else np.zeros(2, dtype=np.float32)
    )

    kp3d = np.full((18, 3), np.nan, dtype=np.float32)
    for i in range(18):
        if np.isfinite(kp2d[i]).all():
            kp3d[i] = [
                 (kp2d[i, 0] - pelvis_px[0]) / ppm,
                -(kp2d[i, 1] - pelvis_px[1]) / ppm,
                0.0,
            ]
    return kp3d, pelvis_px, float(ppm)


class RGBBody18Stream:

    def __init__(
        self,
        source        = 0,
        ref_height_m  : float = 1.7,
        conf_thresh   : float = 0.4,
        tracker       : str   = "bytetrack.yaml",
        device        : str   = "cpu",
        imgsz         : int   = 320,
        enable_view   : bool  = True,
    ):
        """
        Args:
            source:       índice de cámara (int) o ruta a video (str)
            ref_height_m: altura asumida de la persona en metros
            conf_thresh:  umbral de confianza de detección YOLO
            tracker:      bytetrack.yaml (default) o botsort.yaml
            device:       'cpu' o 'cuda'
            imgsz:        tamaño de entrada de YOLO (320=rápido, 640=preciso)
            enable_view:  mostrar ventana "RGB Tracking"
        """
        self.source      = source
        self.ref_height_m = ref_height_m
        self.conf_thresh = conf_thresh
        self.tracker     = tracker
        self.device      = device
        self.imgsz       = imgsz
        self.enable_view = enable_view
        self._cap        = None
        self._yolo       = None

    def open(self):
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"No se pudo abrir: {self.source}")

        print(f"[RGBBody18Stream] Cargando {_YOLO_POSE_MODEL} "
              f"(device={self.device}, imgsz={self.imgsz})...")
        self._yolo = YOLO(_YOLO_POSE_MODEL)

        # Prewarm: la primera inferencia es ~3× más lenta por compilación JIT
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self._yolo(dummy, device=self.device, imgsz=self.imgsz, verbose=False)
        print("[RGBBody18Stream] Listo.")

    def close(self):
        if self._cap is not None:
            self._cap.release()
        cv2.destroyAllWindows()

    def frames(self):
        """
        Generador: devuelve un dict por frame.
        Sale cuando el video termina o el usuario presiona 'q'/'ESC'.
        """
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            # ── YOLO-pose: una llamada da detecciones + tracking + keypoints ──
            t0 = time.perf_counter()
            results = self._yolo.track(
                source   = frame,
                persist  = True,
                tracker  = self.tracker,
                conf     = self.conf_thresh,
                device   = self.device,
                imgsz    = self.imgsz,
                verbose  = False,
            )
            yolo_ms = (time.perf_counter() - t0) * 1000

            result    = results[0]
            annotated = result.plot()
            bodies    = []

            boxes = result.boxes
            kps   = result.keypoints

            if boxes is not None and kps is not None and len(boxes):
                ids      = boxes.id
                xyxys    = boxes.xyxy.cpu().numpy()
                det_conf = boxes.conf.cpu().numpy()
                kp_data  = kps.data.cpu().numpy()   # (N, 17, 3)

                for i in range(len(xyxys)):
                    _, y1, _, y2 = xyxys[i]
                    track_id = int(ids[i].item()) if ids is not None else i

                    kp2d, kp_conf              = _yolo_to_body18_2d(kp_data[i])
                    kp3d, pelvis_px, ppm       = _to_pseudo3d(
                        kp2d, float(y2 - y1), self.ref_height_m)

                    bodies.append({
                        "kp3d":           kp3d,
                        "kp_conf":        kp_conf,
                        "id":             track_id,
                        "confidence":     float(det_conf[i]) * 100.0,
                        "tracking_state": "ok",
                        "pelvis_px":      pelvis_px,   # (2,) píxeles para back-proyección
                        "ppm":            ppm,         # píxeles/metro
                    })

            yield {"image_bgr": annotated, "bodies": bodies, "yolo_ms": yolo_ms}

            if self.enable_view:
                cv2.imshow("RGB Tracking", annotated)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
