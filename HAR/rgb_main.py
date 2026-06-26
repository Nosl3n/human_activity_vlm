"""
Pipeline RGB de estimación de pose + inferencia activa.
Sin dependencia de ZED. Cada persona detectada tiene su propio estimador
(estado mu (16,), Lambda (16,16) independiente por tracking ID).
"""

import argparse
import time
import cv2
import numpy as np

from rgb_body_stream import RGBBody18Stream
from human_kinematic_model import lengths_from_standard_np, HumanKinematicModel, KP18_NAMES
from vfe_inference import AInfLaplacePoseEstimator, InferenceConfig
from visualization_compare import render_camera_overlay, render_unified_panel

# Nombres de ventana — ASCII puro para evitar problemas con GTK en Linux
_WIN_OVERLAY = "HAR - Modelo VFE"
_WIN_PANEL   = "HAR - Panel canonico"


def _template_np() -> np.ndarray:
    """Esqueleto de referencia en metros (Y-up, persona de ≈1.70 m)."""
    kp = np.zeros((18, 3), dtype=np.float32)
    kp[0]  = [ 0.00, 1.65, 0.0]   # nariz
    kp[14] = [ 0.08, 1.68, 0.05]  # ojo der
    kp[15] = [-0.08, 1.68, 0.05]  # ojo izq
    kp[16] = [ 0.12, 1.65, 0.0]   # oreja der
    kp[17] = [-0.12, 1.65, 0.0]   # oreja izq
    kp[1]  = [ 0.00, 1.50, 0.0]   # cuello
    kp[2]  = [ 0.20, 1.50, 0.0]   # hombro der
    kp[5]  = [-0.20, 1.50, 0.0]   # hombro izq
    kp[3]  = [ 0.35, 1.20, 0.0]   # codo der
    kp[4]  = [ 0.50, 0.95, 0.0]   # muñeca der
    kp[6]  = [-0.35, 1.20, 0.0]   # codo izq
    kp[7]  = [-0.50, 0.95, 0.0]   # muñeca izq
    kp[8]  = [ 0.15, 1.00, 0.0]   # cadera der
    kp[11] = [-0.15, 1.00, 0.0]   # cadera izq
    kp[9]  = [ 0.15, 0.60, 0.0]   # rodilla der
    kp[10] = [ 0.15, 0.10, 0.0]   # tobillo der
    kp[12] = [-0.15, 0.60, 0.0]   # rodilla izq
    kp[13] = [-0.15, 0.10, 0.0]   # tobillo izq
    return kp


def _print_summary(pid, conf, mean_l2, rmse, anchors, valid, tr_cov, scale):
    print(f"\n{'='*118}")
    print(f"  Person {pid} | conf={conf:.1f} | mean_L2={mean_l2:.4f}m | RMSE={rmse:.4f}m | "
          f"valid={valid}/18 | tr(cov)={tr_cov:.5f} | scale={scale:.3f} | anchors={anchors}")
    print(f"{'='*118}")


def _print_table(live, pred, diff, per_l2):
    hdr = (f"{'kp':>2}  {'name':<12} | {'lx':>7} {'ly':>7} {'lz':>7} | "
           f"{'px':>7} {'py':>7} {'pz':>7} | {'dx':>7} {'dy':>7} {'dz':>7} | {'L2':>7}")
    print(hdr)
    print("-" * len(hdr))
    for i, name in enumerate(KP18_NAMES):
        def f(v): return f"{v:7.3f}" if np.isfinite(v) else "    nan"
        def fe(v): return f"{v:7.4f}" if np.isfinite(v) else "    nan"
        lx, ly, lz = live[i]
        px, py, pz = pred[i]
        dx, dy, dz = diff[i]
        print(f"{i:2}  {name:<12} | {f(lx)} {f(ly)} {f(lz)} | "
              f"{f(px)} {f(py)} {f(pz)} | {f(dx)} {f(dy)} {f(dz)} | {fe(per_l2[i])}")


def _init_windows(show_panel: bool, panel_w: int, panel_h: int) -> None:
    """
    Crea las ventanas OpenCV UNA SOLA VEZ antes del bucle principal.
    WINDOW_NORMAL permite redimensionar y evita que GTK recree la ventana
    en cada imshow cuando el contenido cambia de tamaño.
    """
    cv2.namedWindow(_WIN_OVERLAY, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_WIN_OVERLAY, 960, 540)   # tamaño inicial razonable

    if show_panel:
        cv2.namedWindow(_WIN_PANEL, cv2.WINDOW_NORMAL)
        # El panel tiene el doble de ancho que la cámara; mostrarlo a la mitad
        cv2.resizeWindow(_WIN_PANEL, min(panel_w // 2, 1280), min(panel_h, 480))


def main():
    p = argparse.ArgumentParser(description="HAR pipeline — camara RGB")

    # Fuente de video
    p.add_argument('--source',      default='0',
                   help='Indice de camara (0,1,...) o ruta a video')
    p.add_argument('--ref_height',  type=float, default=1.7,
                   help='Altura asumida de persona en metros')
    p.add_argument('--det_conf',    type=float, default=0.4,
                   help='Umbral de confianza de deteccion YOLO')
    p.add_argument('--tracker',     default='bytetrack.yaml')
    p.add_argument('--imgsz',       type=int,   default=320,
                   help='Tamano de entrada YOLO-pose (320=rapido, 640=preciso)')
    p.add_argument('--no_view',     action='store_true',
                   help='Desactiva ventanas de visualizacion')
    p.add_argument('--debug_panel', action='store_true',
                   help='Muestra el panel canonico de depuracion (YOLO vs modelo interno)')
    p.add_argument('--output_video', default='',
                   help='Ruta para guardar video de salida (ej: out.mp4)')
    p.add_argument('--print_every', type=int, default=1)

    # Estimador Active Inference
    p.add_argument('--anchors',     default='1,2,5,8,11',
                   help='Articulaciones usadas para alineacion Kabsch')
    p.add_argument('--device',      default='cpu')
    p.add_argument('--sigma_obs',   type=float, default=0.06,
                   help='Ruido de observacion base (m)')
    p.add_argument('--sigma_dyn',   type=float, default=0.25,
                   help='Ruido dinamico (velocidad de cambio de angulos por frame)')
    p.add_argument('--gn_steps',    type=int,   default=1,
                   help='Pasos Gauss-Newton por frame (1=rapido, 2=mejor calidad)')
    p.add_argument('--damping',     type=float, default=1e-3)
    p.add_argument('--w_limits',    type=float, default=5.0,
                   help='Peso del prior de limites articulares')
    p.add_argument('--w_sym',       type=float, default=1.0,
                   help='Peso del prior de simetria bilateral')
    p.add_argument('--sigma_min',   type=float, default=0.02)
    p.add_argument('--sigma_max',   type=float, default=0.15)
    p.add_argument('--min_valid',   type=int,   default=12,
                   help='Minimo de articulaciones validas para procesar')
    p.add_argument('--uncertainty_thresh', type=float, default=0.05,
                   help='Umbral de tr(Sigma) para avisar de alta incertidumbre')

    opt = p.parse_args()
    source  = int(opt.source) if opt.source.isdigit() else opt.source
    anchors = [int(x) for x in opt.anchors.split(",") if x.strip()]

    # ── Modelo cinemático ─────────────────────────────────────────────────────
    lengths   = lengths_from_standard_np(_template_np())
    kin_model = HumanKinematicModel(lengths, device=opt.device).to(opt.device)
    cfg = InferenceConfig(
        anchors=anchors,      device=opt.device,
        sigma_obs=opt.sigma_obs, sigma_dyn=opt.sigma_dyn,
        sigma_min=opt.sigma_min, sigma_max=opt.sigma_max,
        w_limits=opt.w_limits,   w_sym=opt.w_sym,
        gn_steps=opt.gn_steps,   damping=opt.damping,
    )

    estimators: dict = {}

    stream = RGBBody18Stream(
        source=source,         ref_height_m=opt.ref_height,
        conf_thresh=opt.det_conf, tracker=opt.tracker,
        device=opt.device,     imgsz=opt.imgsz,
        enable_view=False,     # el overlay de abajo ya muestra el frame
    )
    stream.open()

    # Crear ventanas UNA sola vez (evita que GTK las recree cada frame)
    if not opt.no_view:
        _init_windows(show_panel=opt.debug_panel, panel_w=1920 * 2, panel_h=1080)
        print("[Vis] Presiona 'q' o ESC para salir.")

    video_writer = None
    frame_idx    = 0
    quit_flag    = False

    try:
        for frame in stream.frames():
            if quit_flag:
                break

            image_bgr = frame["image_bgr"]
            yolo_ms   = frame["yolo_ms"]

            if video_writer is None and image_bgr is not None and opt.output_video:
                h, w = image_bgr.shape[:2]
                video_writer = cv2.VideoWriter(
                    opt.output_video, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w, h)
                )
                print(f"[Video] Guardando en {opt.output_video}")

            frame_idx   += 1
            frame_start  = time.perf_counter()

            all_live:      list = []
            all_pred:      list = []
            all_ids:       list = []
            all_pelvis_px: list = []
            all_ppms:      list = []
            all_scales:    list = []
            all_rmse:      list = []
            total_infer_ms = 0.0

            for body in frame["bodies"]:
                if body["tracking_state"] != "ok":
                    continue

                tid        = body["id"]
                live       = body["kp3d"].astype(np.float32)
                kp_conf    = body.get("kp_conf")
                pelvis_px  = body.get("pelvis_px", np.zeros(2, np.float32))
                ppm        = float(body.get("ppm", 1.0))

                if tid not in estimators:
                    estimators[tid] = AInfLaplacePoseEstimator(kin_model, cfg)

                t0  = time.perf_counter()
                res = estimators[tid].infer(live, kp_conf_np=kp_conf)
                total_infer_ms += (time.perf_counter() - t0) * 1000

                scale = float(np.exp(res.mu[-1]))

                all_live.append(live)
                all_pred.append(res.kp_pred_aligned)
                all_ids.append(tid)
                all_pelvis_px.append(pelvis_px)
                all_ppms.append(ppm)
                all_scales.append(scale)
                all_rmse.append(res.rmse)

                if frame_idx % opt.print_every == 0:
                    _print_summary(tid, body["confidence"],
                                   res.mean_l2, res.rmse,
                                   res.used_anchors, res.valid_count,
                                   res.uncertainty_trace, scale)
                    _print_table(live, res.kp_pred_aligned, res.diff, res.per_l2)

                if res.valid_count < opt.min_valid or \
                        res.uncertainty_trace > opt.uncertainty_thresh:
                    print(f"  [!] Person {tid}: alta incertidumbre — "
                          f"valid={res.valid_count}/18, "
                          f"tr(cov)={res.uncertainty_trace:.4f}")

            # ── Visualización ─────────────────────────────────────────────────
            if not opt.no_view and image_bgr is not None:
                if all_live:
                    overlay = render_camera_overlay(
                        image_bgr,
                        all_live, all_pred,
                        all_pelvis_px, all_ppms,
                        track_ids=all_ids,
                        scale_list=all_scales,
                        rmse_list=all_rmse,
                    )
                    if opt.debug_panel:
                        panel = render_unified_panel(
                            image_bgr, all_live, all_pred, track_ids=all_ids
                        )
                        cv2.imshow(_WIN_PANEL, panel)
                else:
                    # Sin personas: mostrar frame YOLO anotado sin modificar
                    overlay = image_bgr

                cv2.imshow(_WIN_OVERLAY, overlay)

                # waitKey SIEMPRE — necesario para que GTK procese eventos
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    quit_flag = True

                if video_writer is not None and all_live:
                    video_writer.write(np.ascontiguousarray(overlay.astype(np.uint8)))

            # ── Timing ───────────────────────────────────────────────────────
            loop_ms = (time.perf_counter() - frame_start) * 1000
            n = len(frame["bodies"])
            print(f"[Frame {frame_idx:4d}] YOLO={yolo_ms:5.1f}ms  "
                  f"VFE={total_infer_ms:6.1f}ms ({n} persona{'s' if n!=1 else ''})  "
                  f"loop={loop_ms:5.1f}ms")

    finally:
        if video_writer is not None:
            video_writer.release()
            print("[Video] Guardado.")
        stream.close()


if __name__ == "__main__":
    main()
