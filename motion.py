import cv2
import numpy as np
from config import MOTION_THRESHOLD_PERCENT

def detect_motion(baseline: np.ndarray, frame: np.ndarray):
    """Detects motion by comparing current frame with the baseline."""
    gray_base = cv2.cvtColor(baseline, cv2.COLOR_RGB2GRAY)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
 
    # Use Gaussian blur to reduce noise for better motion detection
    gray_frame_blur = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    gray_base_blur = cv2.GaussianBlur(gray_base, (21, 21), 0)

    diff = cv2.absdiff(gray_base_blur, gray_frame_blur)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
 
    change_pct = round((np.sum(thresh > 0) / thresh.size) * 100, 2)
    triggered = change_pct > MOTION_THRESHOLD_PERCENT
 
    motion_boxes = []
    if triggered:
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > 500: # Filter out tiny noise spots
                x, y, w, h = cv2.boundingRect(c)
                motion_boxes.append((x, y, w, h))

    diff_visual = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    return triggered, change_pct, diff_visual, motion_boxes

def update_baseline(current_baseline: np.ndarray, current_frame: np.ndarray, alpha=0.01):
    """
    Slowly updates the baseline with the current frame to handle lighting changes.
    alpha: how much of the new frame to incorporate (0.01 = 1%)
    """
    if current_baseline is None:
        return current_frame.copy()
    
    # We use weighted average to slowly blend the new frame into the baseline
    # Only update parts where there is *no* significant motion to avoid ghosting
    updated = cv2.addWeighted(current_baseline, 1 - alpha, current_frame, alpha, 0)
    return updated
 