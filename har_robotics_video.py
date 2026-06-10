import cv2
import ollama
import os
import time

# 1. Configurar el prompt del sistema (Instrucciones estrictas para el VLM)
SYSTEM_PROMPT = (
    "Eres el sistema de percepción visual de un robot de servicio. Tu tarea es analizar la "
    "imagen y describir la actividad humana orientada en la agricultura en una sola frase corta de máximo 5 palabras. "
    "Sé directo y objetivo. Ejemplos: 'Caminando hacia la salida', 'Escribiendo en la laptop', "
    "'Bebiendo agua', 'Manipulando un objeto'."
)

def formatear_tiempo(milisegundos):
    """Convierte milisegundos a formato hh:mm:ss"""
    segundos_totales = int(milisegundos / 1000)
    horas = segundos_totales // 3600
    minutos = (segundos_totales % 3600) // 60
    segundos = segundos_totales % 60
    return f"{horas:02d}:{minutos:02d}:{segundos:02d}"

def main():
    # --- CONFIGURACIÓN ---
    # Coloca aquí el nombre exacto de tu video (debe estar en la misma carpeta)
    video_name = "agriculture.mp4" 
    log_name = "registro_actividades.txt"

    if not os.path.exists(video_name):
        print(f"Error: No se encontró el archivo de video '{video_name}' en esta carpeta.")
        print("Por favor, guarda tu video aquí con ese nombre o edita el script.")
        return

    # 2. Inicializar la captura del archivo de video
    cap = cv2.VideoCapture(video_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Procesando video: {video_name} ({fps:.2f} FPS)")
    print(f"Los resultados se guardarán en: {log_name}")
    print("Presiona 'q' para cancelar en cualquier momento.")

    # Crear o vaciar el archivo de texto escribiendo la cabecera
    with open(log_name, "w", encoding="utf-8") as f:
        f.write("--- REGISTRO DE ACTIVIDADES HUMANAS (VLM) ---\n")
        f.write(f"Video procesado: {video_name}\n")
        f.write("Tiempo (hh:mm:ss) | Actividad Detectada\n")
        f.write("-----------------------------------------\n")

    last_inference_time_ms = -3000  # Forzar la primera inferencia en el segundo 0
    actividad_detectada = "Inicializando..."

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nFin del video o no se pudo leer el frame.")
            break

        # Obtener el tiempo actual del video en milisegundos (Tiempo real interno del video)
        current_video_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        # 3. Controlar el muestreo (Inferencia cada 3000 ms = 3 segundos de tiempo de video)
        if current_video_time_ms - last_inference_time_ms >= 3000.0:
            timestamp_formateado = formatear_tiempo(current_video_time_ms)
            
            # Guardar temporalmente el frame actual como imagen
            img_path = "current_frame.jpg"
            cv2.imwrite(img_path, frame)
            
            try:
                # 4. Enviar la imagen del video al VLM (Ollama)
                response = ollama.generate(
                    model='llava',
                    prompt=SYSTEM_PROMPT,
                    images=[img_path]
                )
                actividad_detectada = response['response'].strip().replace('\n', ' ')
                
                # Imprimir en la consola para ver el progreso en tiempo real
                log_line = f"[{timestamp_formateado}] -> {actividad_detectada}"
                print(log_line)
                
                # 5. Guardar el resultado en el archivo TXT de forma persistente
                with open(log_name, "a", encoding="utf-8") as f:
                    f.write(f"{timestamp_formateado} | {actividad_detectada}\n")
                
            except Exception as e:
                print(f"[ERROR VLM en {timestamp_formateado}]: {e}")
            
            # Limpiar de forma segura la imagen temporal
            if os.path.exists(img_path):
                os.remove(img_path)
                
            last_inference_time_ms = current_video_time_ms

        # 6. Redimensionar visualización por si el video original es muy grande (4K o Full HD)
        h, w = frame.shape[:2]
        if w > 1280:
            frame_resized = cv2.resize(frame, (1280, int(h * (1280 / w))))
        else:
            frame_resized = frame

        # Dibujar la última actividad detectada sobre el reproductor de video
        cv2.putText(frame_resized, f"ACTIVIDAD: {actividad_detectada}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("Procesando Video - Human Activity Recognition", frame_resized)

        # Cancelar con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nProcesamiento cancelado por el usuario.")
            break

    # Cierre y liberación segura de recursos
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nProcesamiento terminado. Puedes revisar el archivo '{log_name}'.")

if __name__ == "__main__":
    main()