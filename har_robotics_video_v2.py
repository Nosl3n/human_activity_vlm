import cv2
import ollama
import os
import time

# 1. Configurar el prompt del sistema (Instrucciones estrictas)
SYSTEM_PROMPT = (
    "Eres un sistema de percepción de robots agrícolas. Analiza la imagen y genera un reporte breve en una sola línea con el siguiente formato: "
    "'ESTADO: [Activo/Inactivo/Pausa] | TAREA: [Descripción de 3-5 palabras] | PERSONAS: [Número] | HERRAMIENTAS: [Lista breve]'. "
    "Si no hay actividad agrícola, reporta 'ESTADO: Inactivo'. No agregues texto introductorio ni explicaciones adicionales."
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
    #video_name = os.path.expanduser("~/Documentos/Repositorios/Video_HAVLM/agriculture.mp4")
    video_name = os.path.expanduser("~/Doctorado/videos_HAR/agriculture.mp4")
    log_name = "registro_actividades.txt"

    if not os.path.exists(video_name):
        print(f"Error: No se encontró '{video_name}'.")
        return

    cap = cv2.VideoCapture(video_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Procesando: {video_name} ({fps:.2f} FPS)")

    # Cabecera del log
    with open(log_name, "w", encoding="utf-8") as f:
        f.write("--- REGISTRO DE ACTIVIDADES (VLM) ---\n")
        f.write(f"Video: {video_name}\n")
        f.write("Tiempo | Actividad (Max 2 palabras) | Tiempo Respuesta (ms)\n")
        f.write("------------------------------------------------------------\n")

    last_inference_time_ms = -3000
    actividad_detectada = "Inicializando..."
    last_response_time_ms = 0

    print("Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nFin del video.")
            break

        current_video_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        # Control de muestreo (cada 3 segundos de video)
        if current_video_time_ms - last_inference_time_ms >= 3000.0:
            timestamp_formateado = formatear_tiempo(current_video_time_ms)
            
            # Guardar frame temporal
            img_path = "current_frame.jpg"
            cv2.imwrite(img_path, frame)
            
            try:
                # --- MEDICIÓN DE TIEMPO ---
                start_time = time.time()
                
                response = ollama.generate(
                    model='llava',
                    prompt=SYSTEM_PROMPT,
                    images=[img_path]
                )
                
                end_time = time.time()
                inference_time_ms = (end_time - start_time) * 1000
                
                # --- LIMPIEZA DE SALIDA (Forzar 2 palabras) ---
                raw_text = response['response'].strip().replace('\n', ' ')
                # Tomamos solo las primeras 2 palabras si el modelo escribió más
                palabras = raw_text.split()
                actividad_detectada = " ".join(palabras[:2])
                
                # --- SALIDA EN CONSOLA ---
                log_line = f"[{timestamp_formateado}] -> {actividad_detectada} (Tiempo: {inference_time_ms:.1f}ms)"
                print(log_line)
                
                # --- GUARDAR EN LOG ---
                with open(log_name, "a", encoding="utf-8") as f:
                    f.write(f"{timestamp_formateado} | {actividad_detectada} | {inference_time_ms:.1f}ms\n")
                
                last_response_time_ms = inference_time_ms

            except Exception as e:
                print(f"[ERROR VLM]: {e}")
            
            # Limpieza de archivo temporal
            if os.path.exists(img_path):
                os.remove(img_path)
            
            last_inference_time_ms = current_video_time_ms

        # Visualización
        h, w = frame.shape[:2]
        if w > 1280:
            frame_resized = cv2.resize(frame, (1280, int(h * (1280 / w))))
        else:
            frame_resized = frame

        # Dibujar texto con tiempo de respuesta
        texto_superior = f"ACTIVIDAD: {actividad_detectada}"
        texto_inferior = f"Tiempo Respuesta: {last_response_time_ms:.0f}ms"
            
        cv2.imshow("Procesando Video", frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nCancelado por usuario.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nRegistro guardado en '{log_name}'.")

if __name__ == "__main__":
    main()