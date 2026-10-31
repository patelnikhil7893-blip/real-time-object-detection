# Real-Time Object Detection (YOLOv8 + OpenCV)

Webcam ya video par real-time 80 COCO objects detect karta hai (person, car, phone, bottle, etc.).

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python detect.py                              # webcam
python detect.py --source video.mp4           # video file
python detect.py --classes person car         # sirf kuch classes
python detect.py --model yolov8s.pt --conf 0.5
python detect.py --save output.mp4            # result save karo
```
Keys: `q` = quit, `s` = screenshot

## Tips
- FPS kam hai? `yolov8n.pt` use karo, `--imgsz 320` try karo, ya GPU (CUDA) wala PyTorch install karo.
- Accuracy chahiye? `yolov8s.pt` ya `yolov8m.pt` use karo.
- Apne custom objects detect karne ke liye `YOLO("best.pt")` se custom-trained model load karo.
