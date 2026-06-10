# Human Activity VLM

Proyecto de reconocimiento de actividades humanas usando Vision Language Models (VLM) con LLaVA y Ollama.

## Requisitos Previos

- Python 3.12 o superior
- pip actualizado
- Acceso a terminal/línea de comandos
- Conexión a internet para descargar modelos

## Instalación

### 1. Preparar el Entorno Virtual

```bash
# Activar el entorno virtual
source vlm_env/bin/activate

# Para desactivarlo cuando termines
deactivate
```

### 2. Actualizar pip e Instalar Dependencias

Dentro del entorno virtual activado:

```bash
# Asegurar que pip esté actualizado
pip install --upgrade pip

# Instalar las librerías esenciales para la cámara y el VLM
pip install opencv-python ollama
```

### 3. Instalar Ollama

Descarga e instala Ollama desde: https://ollama.com/library/llava

```bash
# En sistemas Linux/macOS
curl -fsSL https://ollama.com/install.sh | sh
```

### 4. Configurar el Servicio de Ollama

```bash
# Iniciar el servicio de Ollama
sudo systemctl start ollama

# (Opcional) Habilitarlo para que se encienda automáticamente
sudo systemctl enable ollama
```

### 5. Verificar la Instalación

```bash
# Verificar el estado del servicio
sudo systemctl status ollama

# Descargar el modelo LLaVA
ollama pull llava

# Verificar los modelos descargados
ollama list
```

## Uso

Una vez completada la instalación y con el servicio de Ollama ejecutándose, puedes ejecutar los scripts del proyecto.
