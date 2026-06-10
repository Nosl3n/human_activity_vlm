import cv2
import ollama
import time
import os

# 1. Configurar el prompt del sistema (Instrucciones estrictas para el VLM)
SYSTEM_PROMPT = (
    "Eres el sistema de percepción visual de un robot de servicio. Tu tarea es analizar la "
    "imagen y describir la actividad humana principal en una sola frase corta de máximo 5 palabras. "
    "Sé directo y objetivo. Ejemplos: 'Caminando hacia la salida', 'Escribiendo en la laptop', "
    "'Bebiendo agua', 'Manipulando un objeto'."
)

def main():
    # 2. Inicializar la cámara web de la laptop (Índice 1 según tu configuración)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara web.")
        return

    print("Sistema HAR iniciado. Presiona 'q' para salir.")
    last_inference_time = time.time()
    actividad_detectada = "Inicializando..."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        
        # 3. Controlar la tasa de refresco (Inferencia cada 3 segundos)
        if current_time - last_inference_time > 3.0:
            # Guardar temporalmente el frame actual como imagen
            img_path = "current_frame.jpg"
            cv2.imwrite(img_path, frame)
            
            try:
                # 4. Enviar la imagen y el prompt a Ollama
                response = ollama.generate(
                    model='llava',
                    prompt=SYSTEM_PROMPT,
                    images=[img_path]
                )
                actividad_detectada = response['response'].strip()
                print(f"[LOG Robot]: Actividad detectada -> {actividad_detectada}")
                
            except Exception as e:
                actividad_detectada = f"Error en inferencia: {e}"
                print(f"[ERROR VLM]: {e}")
            
            # --- CORRECCIÓN AQUÍ ---
            # Se usa os.remove() directamente, que es la función correcta del sistema de archivos.
            if os.path.exists(img_path):
                os.remove(img_path)
                
            last_inference_time = current_time

        # 5. Visualización: Mostrar el resultado sobre el video en tiempo real
        cv2.putText(frame, f"ACTIVIDAD: {actividad_detectada}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("Robot Perception - Human Activity Recognition", frame)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cierre seguro de recursos para evitar errores de ioctl / Bad file descriptor
    cap.release()
    cv2.destroyAllWindows()
    print("Sistema HAR cerrado correctamente.")

if __name__ == "__main__":
    main()