from ultralytics import YOLO
import cv2
import math

# ==============================
# Funciones auxiliares
# ==============================

def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# ==============================
# Memoria lógica
# ==============================

memory = {}   # ID lógico -> centro
MAX_DISTANCE = 80  # umbral de asociación (ajústalo según escena)

# ==============================
# Modelo
# ==============================

model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        source=frame,
        persist=True,
        tracker="my_bytetrack.yaml",
        classes=[0]
    )

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        new_memory = {}

        for box, track_id in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box)

            center_new = get_center((x1, y1, x2, y2))

            assigned_id = track_id  # ID por defecto

            # ==============================
            # RE-ASIGNACIÓN LÓGICA
            # ==============================
            for mem_id, mem_center in memory.items():
                if distance(center_new, mem_center) < MAX_DISTANCE:
                    assigned_id = mem_id
                    break

            # Guardamos en nueva memoria
            new_memory[assigned_id] = center_new

            # ==============================
            # Dibujo
            # ==============================
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {assigned_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Actualizar memoria
        memory = new_memory.copy()

    cv2.imshow("Tracking + Memoria", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()