import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import time
import datetime

# ---------------- CONFIG ----------------
VIDEO_PATH = os.path.expanduser("~/Doctorado/videos_HAR/agriculture_v1.mp4")
OUTPUT_FILE = "resultados_clip.txt"
FRAME_INTERVAL = 30

labels = [
    "Las personas estan trabajando",
    "Una persona que esta trabajando",
    "la persona esta caminando",
    "las personas estan caminando",
    "la persona esta parada",
    "las personas estan paradas",
    "la persona no esta trabajando",
    "las personas no estan trabajando"
]

# ---------------- LOAD MODEL ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ---------------- VIDEO ----------------
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("[ERROR] No se pudo abrir el video")
    exit()

frame_count = 0
segment_id = 0

with open(OUTPUT_FILE, "w") as f:

    #ahora guardamos ID de clase
    f.write("SEG | tiempo(s) | class_id | confianza | latencia(ms)\n")

    print("Procesando video...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_INTERVAL == 0:

            #  tiempo REAL del video
            video_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            video_time_str = str(datetime.timedelta(seconds=int(video_time_sec)))

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            inputs = processor(
                text=labels,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(device)

            # medir latencia
            t0 = time.time()

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits_per_image
                probs = logits.softmax(dim=1)

            t1 = time.time()
            inference_time = (t1 - t0) * 1000

            #  resultado MULTICLASE
            best_idx = probs.argmax().item()
            confidence = probs[0][best_idx].item()

            #  guardar SOLO ID
            f.write(f"{segment_id} | {video_time_str} | {best_idx} | {confidence:.3f} | {inference_time:.1f}\n")

            # debug opcional
            print(f"[SEG {segment_id}] t={video_time_str} → ID={best_idx} ({labels[best_idx]}) | conf={confidence:.3f} | latency={inference_time:.1f} ms")

            segment_id += 1

        frame_count += 1

cap.release()

print(f"\n Finalizado. Resultados guardados en: {OUTPUT_FILE}")