# HomeWatch AI Pro - System Documentation

This document provides an overview of the event detection system, the available event types, and the integration of Telegram alerts and AI-powered scene descriptions.

## 1. System Architecture

The system operates using three primary background threads to ensure real-time performance without blocking the UI:

- **Camera Capture Worker (`app.py`):** Fetches raw frames from the IP camera, performs fast motion detection, and monitors system health (connectivity and obstruction).
- **AI Processing Worker (`app.py`):** Runs the YOLOv8 model on frames where motion was detected to identify specific objects (e.g., people). It manages the event state machine.
- **Flask Web Server:** Serves the dashboard and provides an API for real-time status updates.

## 2. Event Types

### A. Security Events (AI-Powered)
These are high-level events managed by the `EventManager` (`events.py`). They are triggered when specific objects are detected.

- **`person_detected`**: 
  - **Trigger**: Detected by YOLOv8 when a person is in the frame.
  - **Stability**: Requires 3 consecutive frames of detection before triggering to avoid false positives.
  - **Action**: Generates an AI scene description using Gemini 2.0 Flash, saves a 5-second video clip (pre-buffered), saves a snapshot image, and sends a Telegram alert via `alerts.py`.
  - **Cooldown**: 30 seconds (configurable) before another alert can be sent.

### B. System & Environmental Events
These are health checks performed by the Camera Capture Worker.

- **Connection Lost**: Triggered when the IP camera stream cannot be reached for more than 2 consecutive attempts.
- **Camera Obstructed (Pitch Black)**: Triggered when the mean brightness of the frame falls below a critical threshold (e.g., lens is covered in the dark or camera is down).
- **Camera Obstructed (Lens Covered)**: Triggered when the image becomes extremely blurry and dark, indicating something is directly blocking the lens.

### C. Physical Motion Events
- **General Motion**: Triggered by `motion.py` when the change percentage between frames exceeds the `MOTION_THRESHOLD_PERCENT`.
- **Note**: General motion acts as a "trigger" for the AI Processing Worker. It does not create a logged event unless a person is detected.

## 3. Alerts & Documentation

### Telegram Alerts (`alerts.py`)
When a `person_detected` event starts:
1. The system captures the current frame as a high-quality JPEG.
2. The `llm.py` module sends the frame and detection labels to Gemini.
3. Gemini returns a natural language description (e.g., *"A person in a red shirt is walking past the front door while carrying a package."*).
4. `alerts.py` sends this description along with the snapshot to the configured Telegram bot.

### Logging (`alert_log.json`)
Every security event is logged with:
- Timestamp
- AI Description
- Detected objects list
- Path to the saved video clip

## 4. Key Files
- `app.py`: Main entry point, thread management, and Flask logic.
- `events.py`: State machine for starting/ending events.
- `detection.py`: YOLOv8 object detection logic.
- `motion.py`: Fast background-subtraction based motion detection.
- `alerts.py`: Telegram bot integration.
- `llm.py`: Gemini-powered image description logic.
