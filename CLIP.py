import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

# ---------------- CONFIG ----------------
VIDEO_PATH = os.path.expanduser("~/Doctorado/videos_HAR/agriculture.mp4")
  # cambia esto
FRAME_INTERVAL = 30        # frames (≈1 seg si video es 30fps)

# Clases
labels = [
    "a person working in an agricultura",
    
    "a person not working in an agricultura"
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

print("Procesando video...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # coger frame cada intervalo
    if frame_count % FRAME_INTERVAL == 0:

        # convertir a PIL
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # preparar input
        inputs = processor(
            text=labels,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=1)

        best_idx = probs.argmax().item()
        confidence = probs[0][best_idx].item()

        # Clasificación final
        result = "WORKING" if best_idx == 0 else "NOT_WORKING"

        print(f"[SEGMENTO {segment_id}] → {result} | confianza = {confidence:.3f}")

        segment_id += 1

    frame_count += 1

cap.release()
print("\nFinalizado.")
