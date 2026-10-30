import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

VIDEO = "input.mp4"
STEP = 5          # har 5th frame
CONFS = [0.2, 0.4, 0.6, 0.8]


def load_frames(path, step):
    cap = cv.VideoCapture(path)
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(f)
        i += 1
    cap.release()
    return frames


def detect(model, frames, conf):
    # average people per frame
    counts = [len(model(f, classes=[0], conf=conf, verbose=False)[0].boxes) for f in frames]
    return float(np.mean(counts))


def bar(labels, values, title, name):
    plt.figure(figsize=(5, 4))
    plt.bar(labels, values)
    plt.ylabel("people detected per frame")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(name, dpi=150)
    plt.close()


frames = load_frames(VIDEO, STEP)
dark = [cv.convertScaleAbs(f, alpha=0.3, beta=0) for f in frames]
n = YOLO("yolo11n.pt")
x = YOLO("yolo11x.pt")

# 1. harder video: original vs darkened
v = [detect(n, frames, 0.5), detect(n, dark, 0.5)]
bar(["original", "darkened"], v, "Test 1: darkened video (yolo11n, conf 0.5)", "chart1_darken.png")

# 2. yolo11n vs yolo11x
v = [detect(n, frames, 0.5), detect(x, frames, 0.5)]
bar(["yolo11n", "yolo11x"], v, "Test 2: models (conf 0.5)", "chart2_models.png")

# 3. confidence sweep
v = [detect(n, frames, c) for c in CONFS]
plt.figure(figsize=(5, 4))
plt.plot(CONFS, v, marker="o")
plt.xticks(CONFS)
plt.xlabel("confidence threshold")
plt.ylabel("people detected per frame")
plt.title("Test 3: confidence sweep (yolo11n)")
plt.tight_layout()
plt.savefig("chart3_confidence.png", dpi=150)
print("done")