from ultralytics import YOLO
import cv2

# Cargar modelo (detección de personas)
model = YOLO("yolov8s.pt")

# Abrir video o cámara
cap = cv2.VideoCapture(0)  # o ruta a video

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Tracking (usa ByteTrack por defecto configurable)
    results = model.track(
        source=frame,
        persist=True,   # mantiene IDs
        #tracker="botsort.yaml",    # bot sort o bytetrack
        tracker="my_bytetrack.yaml",
        classes=[0]     # SOLO personas (muy importante)
    )

    # Visualización
    annotated_frame = results[0].plot()

    cv2.imshow("Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows() 