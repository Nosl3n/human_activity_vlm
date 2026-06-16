import os
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
    img_resized = cv2.resize(img, (192, 192))
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_img = np.expand_dims(img_resized, axis=0)
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
video_name = os.path.expanduser("~/Doctorado/videos_HAR/recolectando.mp4")
cap = cv2.VideoCapture(video_name)
#cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === TRACKING (solo visual)
    results = model.track(
        source=frame,
        persist=True,
        tracker="my_bytetrack.yaml",
        classes=[0]
    )

    annotated_frame = results[0].plot()

    # ==============================
    # POSE SOBRE TODA LA IMAGEN
    # ==============================
    keypoints = get_pose(frame)
    kp = keypoints[0][0]  # (17,3)

    h, w, _ = frame.shape
    keypoints_px = []

    for k in kp:
        ky, kx, conf = k
        px = int(kx * w)
        py = int(ky * h)
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