import argparse
import cv2 as cv
from ultralytics import YOLO


def draw_box(image, x1, y1, x2, y2, name, conf):
    cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.putText(image, f"{name} {conf:.2f}", (x1, max(y1 - 8, 15)),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="input.mp4")
    p.add_argument("--output", default="output.mp4")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    model = YOLO("yolo11n.pt")
    cap = cv.VideoCapture(args.input)
    fps = cap.get(cv.CAP_PROP_FPS) or 30
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    out = cv.VideoWriter(args.output, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model(frame, classes=[0], conf=args.conf, verbose=False)[0]
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            draw_box(frame, x1, y1, x2, y2,
                     model.names[int(box.cls[0])], float(box.conf[0]))
        out.write(frame)
        if args.show:
            cv.imshow("Result", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()