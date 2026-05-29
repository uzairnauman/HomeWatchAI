import os
import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


def send_alert(
    img_path: str,
    description: str,
    detections: list,
    title: str = "Security Alert",
    zone_text: str | None = None,
    identity_text: str | None = None,
):
    print("DEBUG: send_alert called")
    print("DEBUG: img_path =", img_path)
    print("DEBUG: token loaded =", bool(TELEGRAM_BOT_TOKEN))
    print("DEBUG: chat id loaded =", bool(TELEGRAM_CHAT_ID))

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured, skipping alert.")
        return

    if not img_path or not os.path.exists(img_path):
        print(f"Telegram image missing: {img_path}")
        return

    try:
        unique_labels = list({d.get("label", "object") for d in detections})
        labels = ", ".join(unique_labels) if unique_labels else "None"

        extra_lines = []
        if identity_text:
            extra_lines.append(f"*Identity:* {identity_text}")
        if zone_text:
            extra_lines.append(f"*Zone:* {zone_text}")

        extra_block = "\n".join(extra_lines)
        if extra_block:
            extra_block += "\n\n"

        caption = (
            f"🚨 *{title}*\n\n"
            f"{description}\n\n"
            f"{extra_block}"
            f"*Detected:* {labels}"
        )

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

        with open(img_path, "rb") as photo:
            response = requests.post(
                url,
                data={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "caption": caption,
                    "parse_mode": "Markdown",
                },
                files={"photo": photo},
                timeout=10,
            )

        print("DEBUG: telegram status =", response.status_code)
        print("DEBUG: telegram response =", response.text)

        if response.status_code != 200:
            print(f"Telegram Error: {response.text}")
        else:
            print("📩 Telegram alert sent successfully")

    except Exception as e:
        print(f"Telegram Alert Error: {e}")