import argparse
import time
from collections import Counter

import cv2
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Real-time object detection (YOLOv8)")
    p.add_argument("--source", default="0", help="camera index ya video file path")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--classes", nargs="*", default=None)
    p.add_argument("--save", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)

    class_ids = None
    if args.classes:
        name_to_id = {v.lower(): k for k, v in model.names.items()}
        class_ids = [name_to_id[c.lower()] for c in args.classes]

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Source open nahi hua: {args.source}")

    writer = None
    fps_smooth, prev = 0.0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                                classes=class_ids, verbose=False)[0]
        annotated = results.plot()

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        fps_smooth = fps if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * fps

        counts = Counter(model.names[int(c)] for c in results.boxes.cls)
        cv2.putText(annotated, f"FPS: {fps_smooth:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        for i, (name, n) in enumerate(counts.most_common(8)):
            cv2.putText(annotated, f"{name}: {n}", (10, 58 + 26 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        if args.save:
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
            writer.write(annotated)

        cv2.imshow("Real-Time Object Detection", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            cv2.imwrite(f"screenshot_{int(time.time())}.jpg", annotated)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()