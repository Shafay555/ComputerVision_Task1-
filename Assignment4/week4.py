import cv2
import math
from ultralytics import YOLO
#For Frame Check 
INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "output.mp4"
FRAMES = [61, 62, 63, 64, 65, 66] 

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(INPUT_VIDEO)
w, h = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

seen_ids = set()       
log = []               
last = {}              
candidates = []        
frame_no = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_no += 1

    r = model.track(frame, persist=True, classes=[0],
                    tracker="bytetrack.yaml", verbose=False)[0]
    annotated = r.plot()   

    ids, centers = [], {}
    if r.boxes.id is not None:
        ids = r.boxes.id.int().tolist()
        for i, b in zip(ids, r.boxes.xywh.tolist()):
            centers[i] = (b[0], b[1])

    
    new_ids = [i for i in ids if i not in seen_ids]
    if new_ids:
        log.append(f"Frame {frame_no}: new ID(s) {new_ids} | visible now: {ids}")
        for n in new_ids:
            cx, cy = centers[n]
            for old, (f, ox, oy) in last.items():
                if old not in centers and 0 < frame_no - f <= 30:
                    d = math.hypot(cx - ox, cy - oy)
                    if d < 150:
                        candidates.append(
                            f"Frame {frame_no}: ID {old} -> {n} "
                            f"(gap {frame_no - f} frames, distance {d:.0f}px)")

    seen_ids.update(ids)
    for i, (cx, cy) in centers.items():
        last[i] = (frame_no, cx, cy)

    
    cv2.putText(annotated, f"Unique IDs so far: {len(seen_ids)}", (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    out.write(annotated)

cap.release()
out.release()

with open("id_log.txt", "w") as f:
    f.write("\n".join(log))
with open("switch_candidates.txt", "w") as f:
    f.write("\n".join(candidates))

print("Total unique IDs:", len(seen_ids))
print(len(candidates), "switch candidates ")


cap2 = cv2.VideoCapture(OUTPUT_VIDEO)
for n in FRAMES:
    cap2.set(cv2.CAP_PROP_POS_FRAMES, n - 1)   
    ok, img = cap2.read()
    if ok:
        cv2.imwrite(f"frame_{n}.png", img)
        print("saved", n)
    else:
        print("Frame not found", n)
cap2.release()