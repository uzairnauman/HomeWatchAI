import cv2
import numpy as np
import requests
import os

BASELINE_PATH = "baseline.npy"

# Global state for maintaining active video capture sessions
_current_capture = None
_current_capture_url = None

def get_frame_from_stream(url: str) -> np.ndarray | None:
    global _current_capture, _current_capture_url
    try:
        # Local webcam handling
        if url.strip().lower() == "webcam":
            if _current_capture is None or _current_capture_url != url:
                if _current_capture is not None:
                    _current_capture.release()
                _current_capture = cv2.VideoCapture(0)
                _current_capture_url = url
            
            # Clear buffered frames to stay real-time
            for _ in range(4):
                _current_capture.grab()
            ret, frame = _current_capture.read()
            if ret and frame is not None:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            _current_capture.release()
            _current_capture = None
            _current_capture_url = None
            return None
            
        # MJPEG Continuous Stream handling (e.g. IP Webcams /video)
        elif any(x in url.lower() for x in ["mjpg", "mjpeg", "video.cgi", "/video"]):
            if _current_capture is None or _current_capture_url != url:
                if _current_capture is not None:
                    _current_capture.release()
                cap = cv2.VideoCapture(url)
                # Ensure minimal buffer size so we only grab the absolute newest frame
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                _current_capture = cap
                _current_capture_url = url
            
            _current_capture.grab()
            ret, frame = _current_capture.read()
            if ret and frame is not None:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            _current_capture.release()
            _current_capture = None
            _current_capture_url = None
            return None
            
        # Standard JPEG Snapshot URL
        else:
            resp = requests.get(url, timeout=3)
            arr = np.frombuffer(resp.content, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None

    except Exception as e:
        print(f"Stream Error: {e}")
        return None

def save_baseline(frame: np.ndarray):
    np.save(BASELINE_PATH, frame)

def load_baseline() -> np.ndarray | None:
    if os.path.exists(BASELINE_PATH):
        return np.load(BASELINE_PATH)
    return None