import cv2
import numpy as np
from typing import List, Optional

BODY18_EDGES = [
    (1, 0),
    (0, 14), (14, 16),
    (0, 15), (15, 17),
    (1, 2),  (2,  3),  (3,  4),
    (1, 5),  (5,  6),  (6,  7),
    (2, 8),  (8,  9),  (9,  10),
    (5, 11), (11, 12), (12, 13),
]

_PERSON_COLORS = [
    (0,   255,   0),   # verde
    (0,   165, 255),   # naranja
    (255,   0, 255),   # magenta
    (0,   255, 255),   # amarillo
    (255, 128,   0),   # azul claro
    (128,   0, 255),   # violeta
]


def project_to_2d(kp3d: np.ndarray, width: int, height: int,
                  scale: float = 0.0) -> np.ndarray:
    """
    Orthographic projection: X-right, Y-up → screen (cx, cy-Y*scale).
    scale=0 (default) auto-sizes so a 1.7 m person fills ~75 % of panel height.
    """
    if scale <= 0:
        scale = height * 0.42      # ≈ 0.42 * H px/m → 1.7 m fills ~71 % of H
    cx = width  // 2
    cy = int(height * 0.80)        # pelvis at 80 % from top
    kp2d = np.full((18, 2), np.nan, dtype=np.float32)
    for i, p in enumerate(kp3d):
        if p is None or len(p) != 3 or not np.isfinite(p).all():
            continue
        x, y, z = p
        kp2d[i] = [cx + x * scale, cy - y * scale]
    return kp2d


def _kp3d_to_camera_px(kp3d: np.ndarray, pelvis_px: np.ndarray,
                        ppm: float) -> np.ndarray:
    """
    Inverse of _to_pseudo3d: map 3-D metric coords back to camera pixel coords.
        px = pelvis_px[0] + x * ppm
        py = pelvis_px[1] - y * ppm   (Y-flip: up → negative pixel row)
    """
    kp2d = np.full((18, 2), np.nan, dtype=np.float32)
    for i in range(18):
        if np.isfinite(kp3d[i]).all():
            kp2d[i, 0] = pelvis_px[0] + kp3d[i, 0] * ppm
            kp2d[i, 1] = pelvis_px[1] - kp3d[i, 1] * ppm
    return kp2d


def draw_body18(image: np.ndarray, keypoints_2d: np.ndarray,
                color=(0, 255, 0)) -> np.ndarray:
    img = image.copy()
    for a, b in BODY18_EDGES:
        if np.isfinite(keypoints_2d[a]).all() and np.isfinite(keypoints_2d[b]).all():
            p0 = tuple(np.round(keypoints_2d[a]).astype(int))
            p1 = tuple(np.round(keypoints_2d[b]).astype(int))
            cv2.line(img, p0, p1, color, 2, cv2.LINE_AA)
    for i in range(18):
        if np.isfinite(keypoints_2d[i]).all():
            p = tuple(np.round(keypoints_2d[i]).astype(int))
            cv2.circle(img, p, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, p, 4, color, 1, cv2.LINE_AA)
    return img


def render_split_view(image_left, live_kp: np.ndarray, pred_kp: np.ndarray,
                      left_label: str = "ZED BODY_18",
                      right_label: str = "Internal Model") -> np.ndarray:
    """Single-person split view (observed | predicted) on a light background."""
    if image_left.ndim == 3 and image_left.shape[2] == 4:
        image_left = cv2.cvtColor(image_left, cv2.COLOR_BGRA2BGR)

    H, W = image_left.shape[:2]
    panel_l = np.full((H, W, 3), 245, dtype=np.uint8)
    panel_r = np.full((H, W, 3), 245, dtype=np.uint8)

    panel_l = draw_body18(panel_l, project_to_2d(live_kp, W, H), color=(0, 255, 0))
    panel_r = draw_body18(panel_r, project_to_2d(pred_kp, W, H), color=(255, 0, 0))

    cv2.putText(panel_l, left_label,  (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(panel_r, right_label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    return np.hstack((panel_l, panel_r))


def render_camera_overlay(image_bgr: np.ndarray,
                          live_kps_list: List[np.ndarray],
                          pred_kps_list: List[np.ndarray],
                          pelvis_pxs: List[np.ndarray],
                          ppms: List[float],
                          track_ids: Optional[List[int]] = None,
                          scale_list: Optional[List[float]] = None,
                          rmse_list: Optional[List[float]] = None) -> np.ndarray:
    """
    Overlays both observed (YOLO) and predicted (VFE) skeletons on the camera frame.

    • Observed  — per-person color (same palette as unified panel)
    • Predicted — cyan-shifted version of the same color (easy to distinguish)

    Uses back-projection via pelvis_px + ppm so joints land exactly where the
    person is in the image, regardless of resolution.
    """
    if image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

    img = image_bgr.copy()
    H, W = img.shape[:2]

    for idx, (live, pred, pel, ppm) in enumerate(
            zip(live_kps_list, pred_kps_list, pelvis_pxs, ppms)):

        obs_color  = _PERSON_COLORS[idx % len(_PERSON_COLORS)]
        # Predicted: mix towards cyan so it's clearly different
        pred_color = (
            min(obs_color[0] + 140, 255),
            min(obs_color[1] + 80,  255),
            255,
        )

        live_px = _kp3d_to_camera_px(live, pel, ppm)
        pred_px = _kp3d_to_camera_px(pred, pel, ppm)

        img = draw_body18(img, live_px,  color=obs_color)
        img = draw_body18(img, pred_px,  color=pred_color)

    # Legend (top-left corner)
    cv2.putText(img, "YOLO-pose", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "Modelo VFE", (10, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Per-person metrics (top-right area)
    for idx in range(len(live_kps_list)):
        pid = track_ids[idx] if (track_ids and idx < len(track_ids)) else idx + 1
        s   = scale_list[idx] if scale_list else None
        r   = rmse_list[idx]  if rmse_list  else None
        parts = [f"P{pid}"]
        if s is not None:
            parts.append(f"s={s:.2f}")
        if r is not None:
            parts.append(f"rmse={r:.3f}m")
        txt = "  ".join(parts)
        color = _PERSON_COLORS[idx % len(_PERSON_COLORS)]
        cv2.putText(img, txt, (W - 280, 28 + idx * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return img


def render_unified_panel(image_ref: np.ndarray,
                         live_kps_list: List[np.ndarray],
                         pred_kps_list: List[np.ndarray],
                         track_ids: Optional[List[int]] = None,
                         left_label:  str = "YOLO-pose",
                         right_label: str = "Modelo interno") -> np.ndarray:
    """
    Single fixed-size panel with two sub-panels (OBSERVED | PREDICTED).

    All persons are overlaid in each sub-panel using a distinct color per person,
    so the window stays compact regardless of the number of detected subjects.
    A color legend at the bottom maps colors to person track IDs.
    """
    if image_ref.ndim == 3 and image_ref.shape[2] == 4:
        image_ref = cv2.cvtColor(image_ref, cv2.COLOR_BGRA2BGR)

    H, W = image_ref.shape[:2]

    panel_l = np.full((H, W, 3), 30, dtype=np.uint8)   # dark background
    panel_r = np.full((H, W, 3), 30, dtype=np.uint8)

    for idx, (live, pred) in enumerate(zip(live_kps_list, pred_kps_list)):
        color   = _PERSON_COLORS[idx % len(_PERSON_COLORS)]
        panel_l = draw_body18(panel_l, project_to_2d(live, W, H), color=color)
        panel_r = draw_body18(panel_r, project_to_2d(pred, W, H), color=color)

    # Header labels
    cv2.putText(panel_l, left_label,  (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(panel_r, right_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (220, 220, 220), 2, cv2.LINE_AA)

    # Per-person color legend at bottom
    for idx in range(len(live_kps_list)):
        color = _PERSON_COLORS[idx % len(_PERSON_COLORS)]
        pid   = track_ids[idx] if (track_ids and idx < len(track_ids)) else idx + 1
        x0    = 10 + idx * 75
        y0    = H - 12
        for panel in (panel_l, panel_r):
            cv2.circle(panel, (x0 + 8, y0 - 6), 6, color, -1, cv2.LINE_AA)
            cv2.putText(panel, f"P{pid}", (x0 + 18, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    return np.hstack((panel_l, panel_r))


def render_multi_view(image_ref: np.ndarray,
                      live_kps_list: List[np.ndarray],
                      pred_kps_list: List[np.ndarray],
                      left_label:  str = "YOLO-pose",
                      right_label: str = "Internal Model") -> np.ndarray:
    """
    Stacked-row view: one [live | pred] row per person.
    Kept for reference; use render_unified_panel for the default pipeline.
    """
    if image_ref.ndim == 3 and image_ref.shape[2] == 4:
        image_ref = cv2.cvtColor(image_ref, cv2.COLOR_BGRA2BGR)

    H, W = image_ref.shape[:2]
    rows = []

    for idx, (live, pred) in enumerate(zip(live_kps_list, pred_kps_list)):
        color   = _PERSON_COLORS[idx % len(_PERSON_COLORS)]
        panel_l = np.full((H, W, 3), 245, dtype=np.uint8)
        panel_r = np.full((H, W, 3), 245, dtype=np.uint8)
        panel_l = draw_body18(panel_l, project_to_2d(live, W, H), color=color)
        panel_r = draw_body18(panel_r, project_to_2d(pred, W, H), color=(255, 0, 0))
        cv2.putText(panel_l, f"{left_label}  P{idx+1}",  (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(panel_r, f"{right_label}  P{idx+1}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        rows.append(np.hstack((panel_l, panel_r)))

    return np.vstack(rows) if rows else np.full((H, W * 2, 3), 245, dtype=np.uint8)
