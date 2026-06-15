from ultralytics import YOLO
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# ==============================
# MoveNet
# ==============================
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = movenet.signatures['serving_default']

def get_pose(img):
    img = cv2.resize(img, (192, 192))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_img = np.expand_dims(img, axis=0)
    input_img = tf.cast(input_img, dtype=tf.int32)

    outputs = movenet(input_img)
    return outputs['output_0'].numpy()

# ==============================
# Conexiones del esqueleto
# ==============================
EDGES = [
    (0,1),(0,2),(1,3),(2,4),
    (0,5),(0,6),(5,7),(7,9),
    (6,8),(8,10),(5,6),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16)
]

# ==============================
# YOLO
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

    # 👇 TU visualización base
    annotated_frame = results[0].plot()

    # ==============================
    # AÑADIMOS POSE SOBRE annotated_frame
    # ==============================
    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box)

            # Expandir bounding box
            margin = 0.2
            w = x2 - x1
            h = y2 - y1

            x1e = max(0, int(x1 - margin * w))
            y1e = max(0, int(y1 - margin * h))
            x2e = int(x2 + margin * w)
            y2e = int(y2 + margin * h)

            person_img = frame[y1e:y2e, x1e:x2e]

            if person_img.shape[0] == 0 or person_img.shape[1] == 0:
                continue

            # ==============================
            # POSE
            # ==============================
            keypoints = get_pose(person_img)
            kp = keypoints[0][0]  # (17,3)

            keypoints_px = []

            for k in kp:
                ky, kx, conf = k
                px = int(kx * (x2e - x1e)) + x1e
                py = int(ky * (y2e - y1e)) + y1e
                keypoints_px.append((px, py, conf))

            # ==============================
            # DIBUJAR KEYPOINTS
            # ==============================
            for (px, py, conf) in keypoints_px:
                if conf > 0.3:
                    cv2.circle(annotated_frame, (px, py), 4, (0, 0, 255), -1)

            # ==============================
            # DIBUJAR ESQUELETO
            # ==============================
            for e in EDGES:
                p1 = keypoints_px[e[0]]
                p2 = keypoints_px[e[1]]

                if p1[2] > 0.3 and p2[2] > 0.3:
                    cv2.line(annotated_frame,
                             (p1[0], p1[1]),
                             (p2[0], p2[1]),
                             (255, 0, 0), 2)

    cv2.imshow("Tracking + Pose", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()