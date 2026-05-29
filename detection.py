import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # downloads automatically on first run (~6MB)

def run_detection(frame: np.ndarray):
    results = model(frame, verbose=False)
    detections = []
    annotated = frame.copy()

    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls)]
            conf = float(box.conf)
            if conf > 0.4:
                # Get the box xyxy coordinates
                xyxy = box.xyxy[0].cpu().numpy()
                detections.append({
                    "label": label, 
                    "confidence": round(conf, 2),
                    "box": [int(x) for x in xyxy]
                })

        annotated = r.plot()[:, :, ::-1]  # BGR to RGB

    return detections, annotated