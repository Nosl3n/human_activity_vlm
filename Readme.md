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

Una vez completada la instalación y con el servicio de Ollama ejecutándose:

```bash
# Activar el entorno virtual
source vlm_env/bin/activate

# Asegúrate de estar en el directorio del proyecto
cd ~/har_robotics_project

# Ejecutar el script
python3 har_robotics.py
```

## Prueba con CLIP

### Paso 1: Añadir Repositorio de Python

```bash
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
```

### Paso 2: Actualizar Sistema

```bash
sudo apt update
```

### Paso 3: Instalar Python 3.10

```bash
sudo apt install python3.10 python3.10-venv
```

Verificar la instalación:

```bash
python3.10 --version
```

Debería mostrar: `Python 3.10.x`

### Paso 4: Crear Entorno Virtual para CLIP

```bash
python3.10 -m venv clip_gpu_env
source clip_gpu_env/bin/activate
```

### Paso 5: Instalar PyTorch Compatible con GPU

```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --index-url https://download.pytorch.org/whl/cu116
```

### Paso 6: Instalar Dependencias de Transformers

Para esta versión de Python y PyTorch:

```bash
pip install transformers==4.30.2
pip install accelerate==0.20.3
pip install "numpy<2"
```

### Paso 7: Instalar Transformers y Herramientas Adicionales

```bash
pip install torch torchvision transformers
```
## IMPLEMNETACION DE YOLO

### Paso 1:instalar YOLOv5s + ByteTrack

pip install typeguard
pip install opencv-python
pip install ultralytics

problemas con la version de pytorch: Tu GPU (GTX 1080 Ti → arquitectura Pascal, compute capability 6.1)

OPCIÓN 1 — Ejecutar en CPU (rápido de probar)

results = model.track(
    source=frame,
    persist=True,
    device="cpu"
)

Instalar PyTorch compatible con tu GPU

pip uninstall torch torchvision torchaudio 
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118

Luego instalar la version correcta de numpy
pip install "numpy<2.0"

LUEGO SE PROCEDE A INSTALAR MOVENET PARA LA POSE

Instalar MoveNet

pip install tensorflow tensorflow-hub

