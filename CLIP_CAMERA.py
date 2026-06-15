import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import time

# ---------------- CONFIG ----------------
FRAME_INTERVAL = 30

labels = [
    "hay una agenda roja",
    "hay un Ipad azul",
    "hay una billetara negra",
    "hay un a agenda gris",
    "hay un boligrafo azul"
]

# ---------------- MODEL ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] No se pudo abrir la cámara")
    exit()

frame_count = 0
segment_id = 0

print("Procesando cámara...\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No se pudo leer frame")
            break

        cv2.imshow("Camara", frame)

        # Inferencia cada N frames
        if frame_count % FRAME_INTERVAL == 0:

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            inputs = processor(
                text=labels,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(device)

            
            start_time = time.time()

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits_per_image
                probs = logits.softmax(dim=1)

            end_time = time.time()
            inference_time = end_time - start_time


            best_idx = probs.argmax().item()
            confidence = probs[0][best_idx].item()

            print(f"\n[FRAME {segment_id}] (Inference Time: {inference_time:.3f}s)")

            for i, label in enumerate(labels):
                conf = probs[0][i].item()
                print(f"  {label} → {conf:.3f}")


#-----------------------------------------------
           # print(f"\n[FRAME {segment_id}]")

            #for i, label in enumerate(labels):
             #   conf = probs[0][i].item()
              #  print(f"  {label} → {conf:.3f}")

            # opcional: seguir mostrando la mejor
            #best_idx = probs.argmax().item()
            #best_label = labels[best_idx]
            #best_conf = probs[0][best_idx].item()

            #print(f"→ PREDICCIÓN FINAL: {best_label} ({best_conf:.3f})")

#------------------------------------------------------------------------------------------
            segment_id += 1

        frame_count += 1

        # 🔥 PARAR CON 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupción manual")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Cámara cerrada correctamente")
