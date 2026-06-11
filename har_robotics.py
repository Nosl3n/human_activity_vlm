import cv2
import ollama
import time
import threading
import queue

SYSTEM_PROMPT = (
    "Eres un sistema de percepción de robots agrícolas. Analiza la imagen y genera un reporte breve en una sola línea con el siguiente formato: "
    "'ESTADO: [Activo/Inactivo/Pausa] | TAREA: [Descripción de 3-5 palabras] | PERSONAS: [Número] | HERRAMIENTAS: [Lista breve]'. "
    "Si no hay actividad agrícola, reporta 'ESTADO: Inactivo'. No agregues texto introductorio ni explicaciones adicionales."
)   

USE_GUI = True

# Cola para frames
frame_queue = queue.Queue(maxsize=1)

# Variable compartida
actividad_detectada = "..."


def thread_camara(cap):
    """Captura frames continuamente"""
    global frame_queue

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Frame inválido")
            continue

        # Mantener solo el frame más reciente
        if not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except:
                pass

        frame_queue.put(frame)


def thread_ia():
    """Inferencia con Ollama sin bloquear cámara"""
    global actividad_detectada

    while True:
        try:
            if frame_queue.empty():
                time.sleep(0.1)
                continue

            frame = frame_queue.get()

            # 🔥 CLAVE: codificar imagen en memoria (NO disco)
            _, buffer = cv2.imencode('.jpg', frame)

            response = ollama.generate(
                model='llava',
                prompt=SYSTEM_PROMPT,
                images=[buffer.tobytes()]  # ✅ sin archivo
            )

            actividad_detectada = response.get('response', 'Sin respuesta').strip()
            print(f"[IA]: {actividad_detectada}")

            time.sleep(3)  # control de frecuencia

        except Exception as e:
            print(f"[ERROR IA]: {e}")
            actividad_detectada = "Error IA"


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] No se pudo abrir cámara")
        return

    # Threads
    t_cam = threading.Thread(target=thread_camara, args=(cap,), daemon=True)
    t_ai = threading.Thread(target=thread_ia, daemon=True)

    t_cam.start()
    t_ai.start()

    print("Sistema HAR PRO iniciado")

    try:
        while True:
            if frame_queue.empty():
                continue

            frame = frame_queue.queue[0]  # último frame

            texto = f"ACTIVIDAD: {actividad_detectada[:50]}"

            if USE_GUI:
                cv2.imshow("HAR PRO", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupción manual")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()