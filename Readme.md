##PARA INGRESAR AL ENTORNO VIRTUAL
source vlm_env/bin/activate

deactivate


###SE USA EL MODELO:

# Asegurar que pip esté actualizado dentro de tu entorno
pip install --upgrade pip

# Instalar las librerías esenciales para la cámara y el VLM
pip install opencv-python ollama

### INSTALAR Y/O DESCARGAR OLLAMA: https://ollama.com/library/llava

curl -fsSL https://ollama.com/install.sh | sh

## VERIFICAR Y ARRANCAR EL SERVICIO CON SYSTEMD

# Iniciar el servicio de Ollama
sudo systemctl start ollama

# (Opcional) Habilitarlo para que se encienda automáticamente al prender la laptop
sudo systemctl enable ollama

## PARA VERIFICAR SI ESTA FUNCIONANDO CORRECTAMENTE

sudo systemctl status ollama 

## Descargar el modelo LLaVA

ollama pull llava

## verificar la descarga

ollama list