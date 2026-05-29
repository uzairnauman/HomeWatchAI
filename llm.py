import google.generativeai as genai
import numpy as np
from PIL import Image
from config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

def describe_scene(frame: np.ndarray, detections: list) -> str:
    try:
        img = Image.fromarray(frame)
        labels = ", ".join(d["label"] for d in detections)
        prompt = (
            f"This is a home security camera image. "
            f"The following objects were detected: {labels}. "
            f"Briefly describe what is happening in 1-2 sentences."
        )
        response = model.generate_content([prompt, img])
        return response.text.strip()
    except Exception as e:
        return f"Description unavailable: {e}"