# 🏠 HomeWatch AI Pro

**HomeWatch AI Pro** is an advanced, real-time home security and monitoring system. It leverages computer vision (YOLOv8) and AI (Gemini 2.0) to detect intruders, analyze scenes, and send instant alerts via Telegram.

![System Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Framework](https://img.shields.io/badge/Framework-Flask-lightgrey)

## 🚀 Key Features

- **AI-Powered Detection**: Uses YOLOv8 to identify people with high accuracy, minimizing false alarms from pets or shadows.
- **Natural Language Descriptions**: Integrates Gemini 2.0 Flash to describe exactly what is happening in the scene (e.g., *"A person in a blue jacket is standing near the porch"*).
- **Instant Telegram Alerts**: Sends snapshots and AI-generated descriptions directly to your phone.
- **Rolling Video Buffer**: Automatically saves a 5-second video clip leading up to any detected event.
- **System Health Monitoring**: Alerts you if the camera is obstructed, covered, or if the connection is lost.
- **Live Dashboard**: A clean web interface to monitor your camera feed in real-time.

## 🛠️ Project Structure

- `app.py`: The heart of the system, managing threads for capture, AI, and the web server.
- `detection.py`: YOLOv8 integration for object recognition.
- `motion.py`: Fast motion detection using background subtraction.
- `events.py`: State machine logic for managing event lifecycles.
- `llm.py`: Gemini AI integration for scene analysis.
- `alerts.py`: Telegram bot notification logic.
- `camera.py`: Utilities for handling the IP camera stream.

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd "Home Project"
```

### 2. Install dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file in the root directory (refer to `env.example`):
```env
IP_WEBCAM_URL=http://your-camera-ip:8080/shot.jpg
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
GEMINI_API_KEY=your_gemini_api_key
```

### 4. Run the application
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5000`.

## 🛡️ Event Logic

The system categorizes activities into:
1. **Security Events**: Confirmed person detection with AI analysis and Telegram alerts.
2. **System Events**: Hardware or connectivity issues (Camera Obstructed, Connection Lost).
3. **Motion Triggers**: Initial physical movement that wakes up the AI for deeper analysis.

Detailed technical documentation can be found in [system_documentation.md](./system_documentation.md).

---

*Developed for smart home security enthusiasts.*

# 🛡️ HomeWatch_AI_Surveillance

**AI-powered real-time surveillance system with object detection, face recognition, instant alerts, and automated event recording.**

---

## 📌 Overview

HomeWatch_AI_Surveillance is an intelligent monitoring system that analyzes live video feeds using AI. It detects people, recognizes known faces, identifies objects, and sends instant alerts with recorded evidence.

This project demonstrates the integration of computer vision, machine learning, and web technologies to build a practical surveillance solution.

---

## 🚀 Key Features

- 🎥 **Real-Time Monitoring**: Continuous video stream processing with live detection overlays
- 👤 **Face Recognition**: Identify known vs unknown individuals
- 📦 **Object Detection**: Detect multiple objects (person, phone, bottle, etc.)
- 🚨 **Instant Alerts**: Telegram notifications with image + event summary
- 🎬 **Event Recording**: Automatic 5-second video clips and snapshots
- 🌐 **Live Dashboard**: Clean UI for monitoring and event tracking

---

## 🧠 System Architecture

Camera Input → Frame Capture → AI Processing  
→ Object Detection (YOLO)  
→ Face Recognition (InsightFace)  
→ Event Detection Logic  
→ Dashboard + Alerts + Storage

---

## 🧰 Tech Stack

- **Language**: Python
- **Backend**: Flask
- **Computer Vision**: OpenCV
- **Object Detection**: YOLOv8
- **Face Recognition**: InsightFace
- **Frontend**: HTML, CSS
- **Alerts**: Telegram Bot API

---

## 📁 Project Structure

- `app.py` → Main application logic
- `detection.py` → Object detection module
- `events.py` → Event system
- `alerts.py` → Telegram alerts
- `camera.py` → Camera handling
- `llm.py` → Scene analysis (optional)

Frontend:
- `setup.html` → Camera setup UI
- `dashboard.html` → Monitoring dashboard
- `enroll.html` → Face enrollment

Folders:
- `alerts/` → Event recordings
- `known_faces/` → Registered identities

---

## ⚙️ Setup Instructions

### 1. Clone the Project

```bash
git clone <your-repo-link>
cd HomeWatch_AI_Surveillance
```

### 2. Create Virtual Environment (Recommended Python 3.11)

```bash
python3.11 -m venv env
source env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create `.env` file:

```env
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
GEMINI_API_KEY=your_api_key
ALERT_COOLDOWN_SECONDS=0
AI_INFERENCE_INTERVAL=0.1
```

### 5. Run Application

```bash
python app.py
```

Open:

http://localhost:5000

---

## 🎥 Usage

1. Open setup page
2. Enter camera source:
   - `webcam` → laptop camera
   - IP URL → external camera
3. Preview feed
4. Launch dashboard
5. Monitor events in real time
6. Receive Telegram alerts

---

## 📩 Sample Alert

🚨 Known Person Detected  
Atharva detected by HomeWatch AI  
Objects: 1 bottle  
Identity: Atharva  
Detected: person, bottle

---

## 🔄 Event Flow

Detection → Event Trigger → Snapshot → Video → Alert → Dashboard Update

---

## ⚠️ Limitations

- Requires good lighting
- CPU inference may cause delay
- Detection depends on model accuracy

---

## 🔮 Future Enhancements

- Multi-camera support
- Cloud deployment (AWS / Render)
- Mobile app integration
- Advanced threat classification

---

## 👨‍💻 Author

Atharva Mahamuni  
MS in Computer Science  
AI & Full-Stack Developer

Uzair Nauman
MS (Data Science)
BS (Accounting and Finance)

---

## 📜 License

Academic and educational use only.

---

## ✅ Summary

HomeWatch_AI_Surveillance demonstrates a practical AI system for real-time monitoring, identity recognition, and automated alerting.
