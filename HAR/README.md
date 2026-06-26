# HAR — Human Activity Recognition Pipeline

Documento personal de referencia. Describe la arquitectura, qué hace cada
archivo y cómo ejecutar todo.

---

## Pipeline en una línea

```
Video/Cámara → YOLO-pose → BODY_18 pseudo-3D → Active Inference (VFE) → pose 3D
```

---

## Flujo completo

```
┌──────────────────────────────────────────────────────────────────────┐
│  FUENTE                                                               │
│                                                                       │
│  RGB / video (rgb_main.py)                                           │
│  RGBBody18Stream                                                     │
│  YOLO-pose → 17 kp COCO 2D (píxeles)                                │
│  → conversión → 18 kp BODY_18 pseudo-3D (metros)                    │
└──────────────────────────────────────────────────────────────────────┘
                              │
              kp3d: (18,3) float32 metros
              kp_conf: (18,) confianza [0-100]
                              │
                              ▼
                    vfe_inference.py
                 AInfLaplacePoseEstimator
                 Active Inference (Laplace)
                 minimiza energía libre F:
                   F = F_likelihood
                     + F_dynamics
                     + w_sym  * F_symmetry
                     + w_limits * F_limits
                              │
                              ▼
               human_kinematic_model.py
               HumanKinematicModel
               forward kinematics BODY_18
               15 ángulos + 1 escala → 18 posiciones 3D
                              │
                              ▼
               visualization_compare.py
               render_camera_overlay()       ← ventana principal
               esqueleto VFE proyectado sobre el frame de cámara
               render_unified_panel()        ← panel de depuración
               panel canónico: observado | predicho
```

---

## Archivos

### `rgb_main.py` — punto de entrada RGB

Orquesta el pipeline completo para cámara RGB. Por frame:

1. Pide el siguiente frame a `RGBBody18Stream.frames()`.
2. Por cada persona detectada:
   - Busca o crea un `AInfLaplacePoseEstimator` para ese tracking ID.
   - Llama `estimator.infer(kp3d, kp_conf)`.
   - Acumula live, predicción y tracking ID.
3. Llama `render_camera_overlay()` → ventana principal con el modelo VFE
   proyectado sobre la imagen real de la cámara. Back-proyección via
   `pelvis_px + ppm` (inverso del lifting pseudo-3D).
4. Llama `render_unified_panel()` → panel canónico de depuración
   (esqueleto abstracto sobre fondo oscuro).
5. Imprime métricas por consola (incluyendo escala adaptada).

Un estimador por persona (dict `estimators[track_id]`) → el estado
`(mu (16,), Lambda (16,16))` no se mezcla entre personas distintas.

---

### `rgb_body_stream.py` — captura + detección + conversión

Hace tres cosas en una sola clase `RGBBody18Stream`:

**1. Captura**: `cv2.VideoCapture` (cámara o archivo).

**2. YOLO-pose**: `yolov8n-pose.pt` detecta personas y estima 17 keypoints
COCO en el frame completo.

**3. Conversión COCO-17 → BODY_18 pseudo-3D**:

  ```
  BODY_18[0]  nose       ← COCO[0]  nose
  BODY_18[1]  neck       ← sintético: promedio de r_shoulder + l_shoulder
  BODY_18[2]  r_shoulder ← COCO[6]
  BODY_18[3]  r_elbow    ← COCO[8]
  BODY_18[4]  r_wrist    ← COCO[10]
  BODY_18[5]  l_shoulder ← COCO[5]
  BODY_18[6]  l_elbow    ← COCO[7]
  BODY_18[7]  l_wrist    ← COCO[9]
  BODY_18[8]  r_hip      ← COCO[12]
  BODY_18[9]  r_knee     ← COCO[14]
  BODY_18[10] r_ankle    ← COCO[16]
  BODY_18[11] l_hip      ← COCO[11]
  BODY_18[12] l_knee     ← COCO[13]
  BODY_18[13] l_ankle    ← COCO[15]
  BODY_18[14] r_eye      ← COCO[2]
  BODY_18[15] l_eye      ← COCO[1]
  BODY_18[16] r_ear      ← COCO[4]
  BODY_18[17] l_ear      ← COCO[3]
  ```

  Lifting pseudo-3D (`_to_pseudo3d`):
  - `ppm = bbox_height_px / ref_height_m` → píxeles por metro
  - Origen en pelvis (promedio de caderas idx 8 y 11)
  - `X = (px − pelvis_x) / ppm`
  - `Y = −(py − pelvis_y) / ppm`  (flip: Y imagen→abajo, modelo Y→arriba)
  - `Z = 0`  (sin profundidad, cámara monocular)

  Joints con confianza < 0.2 quedan como NaN.

---

### `vfe_inference.py` — estimador Active Inference

`AInfLaplacePoseEstimator` mantiene `q(θ) ~ N(μ, Σ)` con **θ ∈ R^16**.

Por frame actualiza μ con pasos Gauss-Newton minimizando la energía libre F.
El gradiente y Jacobiano se calculan con autodiferenciación de PyTorch.

**Los 16 parámetros θ:**

| # | Nombre | Descripción |
|---|--------|-------------|
| 0-2 | `sh_L` | Hombro izquierdo — vector axis-angle 3D (Rodrigues) |
| 3-5 | `sh_R` | Hombro derecho — vector axis-angle 3D (Rodrigues) |
| 6 | `el_L` | Flexión codo izquierdo |
| 7 | `el_R` | Flexión codo derecho |
| 8 | `lb_x` | Traslación pelvis en X |
| 9 | `lb_z` | Traslación pelvis en Z |
| 10 | `lb_roll` | Rotación Y del tren inferior |
| 11 | `hip_L_flex` | Flexión cadera izquierda (sagital) |
| 12 | `hip_R_flex` | Flexión cadera derecha (sagital) |
| 13 | `knee_L_flex` | Flexión rodilla izquierda |
| 14 | `knee_R_flex` | Flexión rodilla derecha |
| 15 | `log_s` | Logaritmo del factor de escala global (s = exp(log_s)) |

**Factor de escala (`log_s`):**
El parámetro 15 adapta las proporciones del template al sujeto real.
`s = exp(log_s)`: con `log_s=0` no hay cambio (s=1.0); límites ±0.5
permiten escalar ≈ 61%–165% de la plantilla. Se imprime como `scale=X.XXX`
en consola. Esto elimina el error sistemático cuando la persona es
notablemente más alta o baja que la plantilla de 1.70 m.

**Alineación Kabsch:**
Después de escalar el esqueleto predicho (`kp * s`), Kabsch estima
(R, t) sobre los joints ancla. Esto desacopla la pose interna (ángulos)
de la posición global en la imagen.

---

### `human_kinematic_model.py` — modelo cinemático BODY_18

`HumanKinematicModel`: módulo PyTorch que dado el dict de ángulos
genera las posiciones 3D de los 18 joints.

**Rotaciones de hombro — Rodrigues (axis-angle):**
Reemplaza las matrices Euler (Ry·Rx·Rz) con la fórmula de Rodrigues:

```
R = I + sin(θ)·K + (1−cos(θ))·K²
```

donde `v = (rx, ry, rz)` es el vector axis-angle, `θ = ||v||`,
`K` es la matriz antisimétrica de `v/θ`. Ventajas:
- Sin gimbal lock.
- Espacio de parámetros sin discontinuidades.
- Gradientes suaves en θ→0 (via `clamp(min=1e-8)`).
- Con μ=0 la postura inicial es siempre la postura neutral (identidad).

**Tren inferior articulado:**
Cada pierna tiene flexión de cadera y rodilla independientes (4 params nuevos):
- `hip_L/R_flex`: rota el muslo respecto al marco local de la pelvis (eje X).
- `knee_L/R_flex`: rota la pantorrilla respecto al muslo (eje X, apilado).

```
pierna izq:
  l_kn = l_hip + R_lb @ (R_hip_L @ v_thigh)
  l_an = l_kn  + R_lb @ (R_hip_L @ (R_knee_L @ v_calf))
```

Esto permite modelar caminar, agacharse y sentarse, que antes producían
solo traslación rígida del bloque de cadera/rodilla/tobillo.

Longitudes de segmento: fijas por sesión (derivadas del esqueleto template).
El parámetro `log_s` en vfe_inference escala el esqueleto completo antes
de la alineación Kabsch.

---

### `visualization_compare.py` — visualización

**`render_camera_overlay(image_bgr, live_list, pred_list, pelvis_pxs, ppms, ...)`**:
Ventana principal del pipeline. Back-proyecta los joints del modelo VFE al
espacio píxel de la cámara y los dibuja sobre el frame real. YOLO en el
color de la persona, VFE en variante cian del mismo color. Leyenda y
métricas (scale, RMSE) en la esquina superior derecha. Permite 'q'/ESC
para salir.

**`render_unified_panel(image_ref, live_list, pred_list, track_ids)`**:
Panel de depuración de tamaño fijo (H × 2W). Sub-panel izquierdo: todos los
keypoints observados solapados (proyección ortográfica canónica).
Sub-panel derecho: todas las predicciones solapadas.
Cada persona con color distinto + leyenda con tracking ID.

**`render_split_view(...)`**: vista single-person (observado | predicho).

**`render_multi_view(...)`**: vista legacy con filas apiladas por persona.
Se conserva para referencia pero no se usa en el pipeline por defecto.

**`project_to_2d(kp3d, W, H)`**: proyección ortográfica para visualización
(no es proyección de cámara real, solo para el panel de depuración).

---

### `mejoras.txt`

Análisis de bottlenecks y estrategias de mejora para velocidad, calidad
de pose y profundidad (Z=0).

---

## Cómo ejecutar

Activar entorno virtual:
```bash
source ~/Doctorado/Entornos_Virtuales/har_env/bin/activate
cd ~/Doctorado/HAR
```

Con video grabado:
```bash
python rgb_main.py --source ~/Doctorado/videos_HAR/recolectando.mp4
```

Con cámara en vivo:
```bash
python rgb_main.py --source 0
```

Presiona `q` o `ESC` en la ventana para salir.

Con panel de depuración (segundo panel canónico YOLO vs. VFE):
```bash
python rgb_main.py --source 0 --debug_panel
```

Sin ventanas (solo consola):
```bash
python rgb_main.py --source ~/Doctorado/videos_HAR/recolectando.mp4 --no_view
```

Guardar video de salida:
```bash
python rgb_main.py --source ~/Doctorado/videos_HAR/recolectando.mp4 \
    --output_video resultado.mp4
```

---

## Argumentos clave

| Argumento | Default | Efecto |
|---|---|---|
| `--source` | `0` | Cámara (número) o ruta a video |
| `--imgsz` | `320` | Resolución YOLO. Más alto = más preciso, más lento |
| `--gn_steps` | `1` | Pasos VFE/frame. 1=rápido (~90ms), 2=mejor calidad (~180ms) |
| `--device` | `cpu` | `cpu` o `cuda` (si hay GPU disponible) |
| `--ref_height` | `1.7` | Altura asumida de persona en metros para escala inicial |
| `--no_view` | False | Desactiva ventanas OpenCV |
| `--debug_panel` | False | Muestra panel canónico de depuración (YOLO \| VFE) además del overlay |
| `--det_conf` | `0.4` | Umbral de confianza de detección YOLO |
| `--anchors` | `1,2,5,8,11` | Joints ancla para alineación Kabsch |
| `--sigma_obs` | `0.06` | Ruido de observación (m): mayor = más suavizado |
| `--sigma_dyn` | `0.25` | Ruido dinámico: mayor = sigue movimientos más rápido |
| `--w_limits` | `5.0` | Peso del prior de límites articulares |
| `--w_sym` | `1.0` | Peso del prior de simetría bilateral (hombros, codos, caderas, rodillas) |

---

## Tiempos medidos (CPU, D=16 parámetros)

| Componente | Tiempo |
|---|---|
| YOLO-pose imgsz=320 | ~43ms |
| YOLO-pose imgsz=640 | ~65ms |
| VFE gn_steps=1 (1 persona) | ~90–100ms |
| VFE gn_steps=2 (1 persona) | ~180–200ms |
| **Total 1 persona (defaults)** | **~135ms → ~7 FPS** |
| Total 2 personas (defaults) | ~225ms → ~4 FPS |

Con GPU (`--device cuda`): estimado ~30ms → ~33 FPS.

> **Nota:** D=16 añade ligera latencia respecto a D=11 anterior
> (~10-15ms por paso GN). Para tiempo real crítico usar `--gn_steps 1`.
