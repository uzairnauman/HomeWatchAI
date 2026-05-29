import os
from dotenv import load_dotenv
 
load_dotenv()
 
# IP Webcam app URL — open the app on your phone, it shows the URL at the bottom
# e.g. http://192.168.1.5:8080
IP_WEBCAM_URL = os.getenv("IP_WEBCAM_URL", "http://192.168.1.5:8080/shot.jpg")
 
ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", "30"))
MOTION_THRESHOLD_PERCENT = float(os.getenv("MOTION_THRESHOLD_PERCENT", "1.5"))
 
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
 