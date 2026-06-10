# Human Activity VLM

Proyecto de reconocimiento de actividades humanas usando Vision Language Models (VLM) con LLaVA y Ollama.

## Requisitos Previos

- Python 3.12 o superior
- pip actualizado
- Acceso a terminal/línea de comandos
- Conexión a internet para descargar modelos

## Instalación

### 1. Instalar Herramientas de Entorno Virtual (Ubuntu)

Ubuntu bloquea la instalación global de paquetes Pip (PEP 668). Instala las herramientas necesarias:

```bash
sudo apt update
sudo apt install python3-venv python3-pip -y
```

### 2. Crear el Entorno Virtual

```bash
mkdir ~/har_robotics_project
cd ~/har_robotics_project
python3 -m venv vlm_env

# Activar el entorno virtual
source vlm_env/bin/activate

# Para desactivarlo cuando termines
deactivate
```

### 3. Actualizar pip e Instalar Dependencias

Dentro del entorno virtual activado:

```bash
# Asegurar que pip esté actualizado
pip install --upgrade pip

# Instalar las librerías esenciales para la cámara y el VLM
pip install opencv-contrib-python-headless ollama
```
sudo apt update
sudo apt install libgtk-3-0 libgtk2.0-0 pkg-config

### 4. Instalar Ollama

Descarga e instala Ollama desde: https://ollama.com/library/llava

```bash
# En sistemas Linux/macOS
curl -fsSL https://ollama.com/install.sh | sh
```

### 5. Configurar el Servicio de Ollama

```bash
# Iniciar el servicio de Ollama
sudo systemctl start ollama

# (Opcional) Habilitarlo para que se encienda automáticamente
sudo systemctl enable ollama
```

### 6. Verificar la Instalación

```bash
# Verificar el estado del servicio
sudo systemctl status ollama

# Descargar el modelo LLaVA
ollama pull llava

# Verificar los modelos descargados
ollama list
```

## Uso

Una vez completada la instalación y con el servicio de Ollama ejecutándose:

```bash
# Asegúrate de estar en el directorio del proyecto y el entorno activado
cd ~/har_robotics_project
source vlm_env/bin/activate

# Ejecutar el script
python3 har_robotics.py
```

## Solución de Problemas

### Advertencias de QFontDatabase (OpenCV en Ubuntu)

Las advertencias `QFontDatabase: Cannot find font directory` son comunes al usar OpenCV en entornos virtuales en Ubuntu 24. No afectan la ejecución del programa pero ensucian los logs.

**Solución:**

```bash
pip install opencv-python
pip install opencv-contrib-python-headless
```
