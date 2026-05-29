import json
import os
import cv2
import numpy as np
from datetime import datetime
 
LOG_FILE = "alert_log.json"
ALERT_IMG_DIR = "alerts"
 
os.makedirs(ALERT_IMG_DIR, exist_ok=True)
 
def save_alert(frame: np.ndarray, description: str, detections: list) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = os.path.join(ALERT_IMG_DIR, f"alert_{timestamp}.jpg")
    cv2.imwrite(img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
 
    log = load_alert_log()
    log.append({
        "timestamp": timestamp,
        "description": description,
        "detections": detections,
        "image": img_path
    })
 
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)
 
    return img_path
 
def load_alert_log() -> list:
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    return []