# Human Activity VLM

Proyecto de reconocimiento de actividades humanas usando Vision Language Models (VLM) con LLaVA y Ollama.

## Requisitos Previos

- Python 3.12 o superior
- pip actualizado
- Acceso a terminal/línea de comandos
- Conexión a internet para descargar modelos

## Guía de Instalación

Sigue estos pasos en orden para configurar el proyecto correctamente.

### Paso 1: Instalar Herramientas del Sistema (Ubuntu)

Ubuntu bloquea la instalación global de paquetes Pip (PEP 668). Instala las herramientas necesarias:

```bash
sudo apt update
sudo apt install python3-venv python3-pip -y
sudo apt install libgtk2.0-dev pkg-config
sudo apt install libgtk-3-0 libgtk2.0-0
```

### Paso 2: Crear el Entorno Virtual

```bash
# Crear el entorno virtual
python3 -m venv vlm_env

# Activar el entorno virtual
source vlm_env/bin/activate

# Para desactivarlo cuando termines
deactivate
```

### Paso 3: Actualizar pip e Instalar Dependencias de Python

Dentro del entorno virtual activado:

```bash
# Asegurar que pip esté actualizado
pip install --upgrade pip

# Instalar las librerías esenciales para la cámara y el VLM
pip install opencv-python
pip install opencv-contrib-python-headless
pip install ollama
```

### Paso 4: Instalar Ollama

Descarga e instala Ollama desde: https://ollama.com/library/llava

```bash
# En sistemas Linux/macOS
curl -fsSL https://ollama.com/install.sh | sh
```

### Paso 5: Configurar el Servicio de Ollama

```bash
# Iniciar el servicio de Ollama
sudo systemctl start ollama

# (Opcional) Habilitarlo para que se encienda automáticamente
sudo systemctl enable ollama
```

### Paso 6: Verificar la Instalación

```bash
# Verificar el estado del servicio
sudo systemctl status ollama

# Descargar el modelo LLaVA
ollama pull llava

# Verificar los modelos descargados
ollama list
```

## Ejecución del Proyecto

### ENFOQUE 1: Vision Language Model (LLaVA)

Requiere el servicio de Ollama ejecutándose:

```bash
# En una terminal, iniciar Ollama
ollama serve

# En otra terminal, activar entorno
source vlm_env/bin/activate

# Análisis en tiempo real (cámara)
python3 har_robotics.py

# Análisis de video
python3 har_robotics_video.py
```

**Características:**
- Análisis semántico con LLaVA
- Genera reportes: ESTADO | TAREA | PERSONAS | HERRAMIENTAS
- Modo GUI y registro en log

---

### ENFOQUE 2: CLIP (Contrastive Learning)

No requiere Ollama. Usa modelos de OpenAI sin GPU es más lento.

```bash
source vlm_env/bin/activate

# Análisis en tiempo real (cámara)
python3 CLIP_CAMERA.py

# Análisis de video
python3 CLIP.py
```

**Características:**
- Clasificación de actividades por similitud semántica
- Etiquetas configurables
- Más ligero que LLaVA

---

### ENFOQUE 3: Detección y Tracking (YOLOv8 + ByteTrack)

Detección de personas y seguimiento sin análisis de actividades.

```bash
source vlm_env/bin/activate

# Detección y tracking en tiempo real
python3 ODT.py

# Estimación de pose
python3 TRAKING_POSE.py
```

**Características:**
- Detección con YOLOv8
- Tracking con ByteTrack o BotSORT
- Estimación de pose con MoveNet
- Solo personas (clase 0 de COCO)

---

## Configuración por Enfoque

### LLaVA: Cambiar video de entrada

En `har_robotics_video.py`:

```python
video_name = os.path.expanduser("~/Doctorado/videos_HAR/agriculture.mp4")
```

### CLIP: Cambiar intervalo de frames y etiquetas

En `CLIP.py`:

```python
VIDEO_PATH = os.path.expanduser("~/Doctorado/videos_HAR/agriculture.mp4")
FRAME_INTERVAL = 30  # Frames a procesar

labels = [
    "a person working in agriculture",
    "a person not working in agriculture"
]
```

### YOLO: Cambiar tracker

En `ODT.py`:

```python
tracker="my_bytetrack.yaml"  # ByteTrack (rápido)
# tracker="botsort.yaml"      # BotSORT (preciso)
```

---

## Salida de Datos

- **LLaVA y CLIP:** `registro_actividades.txt` - Log de actividades detectadas
- **YOLO:** Visualización en tiempo real en ventana

---

## Requisitos de Hardware

- **CPU mínima:** Intel i7 / AMD Ryzen 5
- **GPU recomendada:** NVIDIA GTX 1650+ (para LLaVA)
- **RAM:** 8GB mínimo (16GB para video 4K)
- **Almacenamiento:** 10GB para modelos

---

## Instalación Específica por Enfoque

### CLIP - Instalación Detallada (Python 3.10 + PyTorch 1.13)

Si tienes problemas de compatibilidad, usa esta guía:

```bash
# Instalar Python 3.10
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-venv

# Crear entorno
python3.10 -m venv clip_gpu_env
source clip_gpu_env/bin/activate

# Instalar PyTorch 1.13 con CUDA 11.6
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --index-url https://download.pytorch.org/whl/cu116

# Dependencias
pip install transformers==4.30.2
pip install accelerate==0.20.3
pip install "numpy<2"
pip install opencv-python pillow
```

---

### YOLO + ByteTrack - Instalación Detallada

```bash
source vlm_env/bin/activate

# Paquetes base
pip install typeguard
pip install opencv-python
pip install ultralytics
```

**Problema con GPU (GTX 1080 Ti - Pascal architecture, compute capability 6.1):**

**Opción 1: Ejecutar en CPU (más compatible)**

En `ODT.py`, agrega:

```python
results = model.track(
    source=frame,
    persist=True,
    device="cpu"  # Usa CPU
)
```

**Opción 2: Instalar PyTorch compatible con tu GPU**

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2.0"
```

---

### MoveNet (Estimación de Pose) - Instalación

```bash
source vlm_env/bin/activate

pip install tensorflow tensorflow-hub

# Si tienes problemas, instala versión específica:
pip install tensorflow==2.13.0 tensorflow-hub==0.14.0
```

---

## Solución de Problemas de Dependencias

| Error | Solución |
|-------|----------|
| `ImportError: No module named torch` | Reinstalar: `pip install torch torchvision` |
| `NumPy version conflict` | `pip install "numpy<2.0"` |
| `CUDA out of memory` | Usar `device="cpu"` en YOLO |
| `TensorFlow no carga modelos` | `pip install tensorflow==2.13.0` |
| `Ollama connection refused` | Ejecutar `ollama serve` en otra terminal |

