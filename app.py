import time
import os
import json
import cv2
import queue
import threading
import collections
import numpy as np
from flask import Flask, Response, render_template, request, redirect, jsonify
from datetime import datetime
from dotenv import load_dotenv

from detection import run_detection
from alerts import send_alert

# ── InsightFace ───────────────────────────────────────────────────────────────
try:
    from insightface.app import FaceAnalysis

    _face_app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
    _face_app.prepare(ctx_id=0, det_size=(320, 320))
    FACE_RECOGNITION_AVAILABLE = True
    print("✅ InsightFace loaded.")
except Exception as e:
    _face_app = None
    FACE_RECOGNITION_AVAILABLE = False
    print(f"⚠️  InsightFace not available: {e}")

load_dotenv()

app = Flask(__name__)

ALERT_IMG_DIR = "alerts"
KNOWN_FACES_DIR = "known_faces"
LOG_FILE = "alert_log.json"

EVENT_RECORD_SECONDS = int(os.getenv("EVENT_RECORD_SECONDS", "5"))
AI_INFERENCE_INTERVAL = float(os.getenv("AI_INFERENCE_INTERVAL", "0.2"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))
EVENT_REARM_SECONDS = float(os.getenv("EVENT_REARM_SECONDS", "0.2"))

os.makedirs(ALERT_IMG_DIR, exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# ── Face registry ─────────────────────────────────────────────────────────────
known_faces: list[dict] = []
known_faces_lock = threading.Lock()


def _slugify(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-6)
    b = b / (np.linalg.norm(b) + 1e-6)
    return float(1.0 - np.dot(a, b))


def load_known_faces():
    global known_faces

    if not FACE_RECOGNITION_AVAILABLE:
        return

    entries = []
    slugs = set()

    for fname in os.listdir(KNOWN_FACES_DIR):
        base, ext = os.path.splitext(fname)
        if ext.lower() in (".jpg", ".jpeg", ".png", ".npz"):
            slugs.add(base)

    for slug in sorted(slugs):
        display_name = slug.replace("_", " ").title()
        npz_path = os.path.join(KNOWN_FACES_DIR, f"{slug}.npz")
        jpg_path = None

        for ext in (".jpg", ".jpeg", ".png"):
            candidate = os.path.join(KNOWN_FACES_DIR, f"{slug}{ext}")
            if os.path.exists(candidate):
                jpg_path = candidate
                break

        embedding = None

        if os.path.exists(npz_path):
            try:
                data = np.load(npz_path)
                embedding = data["embedding"]
                count = int(data["sample_count"]) if "sample_count" in data else "?"
                print(f"  ✅ Loaded (npz, {count} samples): {display_name}")
            except Exception as e:
                print(f"  ⚠️ Failed to load {npz_path}: {e}")

        if embedding is None and jpg_path:
            try:
                img = cv2.imread(jpg_path)
                if img is not None:
                    faces = _face_app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    if faces:
                        embedding = faces[0].normed_embedding
                        print(f"  ✅ Loaded (jpg inference): {display_name}")
                    else:
                        print(f"  ⚠️ No face found in {jpg_path}, skipping.")
            except Exception as e:
                print(f"  ⚠️ Failed to process {jpg_path}: {e}")

        if embedding is not None:
            entries.append({"name": display_name, "embedding": embedding})

    with known_faces_lock:
        known_faces = entries

    print(f"👤 {len(entries)} known face(s) loaded.")


def identify_faces(rgb_frame: np.ndarray) -> list[dict]:
    if not FACE_RECOGNITION_AVAILABLE:
        return []

    try:
        faces = _face_app.get(rgb_frame)
    except Exception:
        return []

    with known_faces_lock:
        registry = list(known_faces)

    results = []
    for face in faces:
        emb = face.normed_embedding
        bbox = face.bbox.astype(int)

        name = "Unknown"
        known = False

        if registry:
            distances = [_cosine_distance(emb, entry["embedding"]) for entry in registry]
            best_idx = int(np.argmin(distances))
            if distances[best_idx] < SIMILARITY_THRESHOLD:
                name = registry[best_idx]["name"]
                known = True

        results.append(
            {
                "name": name,
                "known": known,
                "box": (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
            }
        )

    return results


# ── Shared state ──────────────────────────────────────────────────────────────
def create_placeholder_frame(text="Connecting..."):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    _, buffer = cv2.imencode(".jpg", frame)
    return buffer.tobytes()


global_frame = create_placeholder_frame("Initializing System...")
frame_lock = threading.Lock()
ai_queue: queue.Queue = queue.Queue(maxsize=1)

frame_buffer: collections.deque = collections.deque(maxlen=150)

latest_detections: list[dict] = []
latest_faces: list[dict] = []
annotation_lock = threading.Lock()

system_event: str | None = None
system_event_time: float = 0.0

# Recording / alert state
recording_active = False
recording_end_time = 0.0
recording_frames: list[np.ndarray] = []
recording_meta: dict | None = None
recording_lock = threading.Lock()

latest_event_payload: dict | None = None
latest_event_lock = threading.Lock()
ui_status = "idle"

last_event_end_time = 0.0
last_alert_signature: dict | None = None

# Enrollment state
_enroll_lock = threading.Lock()
_enroll_state = {
    "active": False,
    "name": "",
    "collected": 0,
    "target": 20,
    "status": "idle",
    "message": "",
}


# ── Persistence ───────────────────────────────────────────────────────────────
def save_fixed_event_video(frames: list[np.ndarray], timestamp: str) -> str:
    vid_path = os.path.join(ALERT_IMG_DIR, f"alert_{timestamp}.mp4")

    if not frames:
        return ""

    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(vid_path, fourcc, 15.0, (w, h))

    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()
    return vid_path


def append_event_log(record: dict):
    log = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE) as f:
                log = json.load(f)
        except Exception:
            log = []

    log.append(record)

    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


# ── Alert helpers ─────────────────────────────────────────────────────────────
def build_alert_message(detections: list, faces: list) -> tuple[str, str, str | None]:
    person_count = sum(1 for d in detections if d.get("label", "").lower() == "person")
    known_names = sorted(
        {
            f["name"]
            for f in faces
            if f.get("known") and f.get("name") not in (None, "", "Unknown")
        }
    )
    unknown_count = sum(1 for f in faces if not f.get("known"))

    labels = []
    counts = {}
    for d in detections:
        label = d.get("label", "object")
        counts[label] = counts.get(label, 0) + 1

    for label, count in sorted(counts.items()):
        if label.lower() != "person":
            labels.append(f"{count} {label}")

    if known_names:
        identity_text = ", ".join(known_names)
        title = "Known Person Detected"
        description = f"{identity_text} detected by HomeWatch AI."
    elif unknown_count > 0:
        identity_text = "Unknown"
        title = "Unidentified Person Incoming"
        description = "Unknown person detected by HomeWatch AI."
    else:
        identity_text = None
        title = "Security Alert"
        if person_count > 0:
            description = f"{person_count} person(s) detected by HomeWatch AI."
        else:
            description = "Security event detected by HomeWatch AI."

    if labels:
        description += " Objects: " + ", ".join(labels) + "."

    return title, description, identity_text


def build_scene_signature(detections: list, faces: list) -> dict:
    object_counts = {}
    for d in detections:
        label = d.get("label", "object").lower()
        object_counts[label] = object_counts.get(label, 0) + 1

    known_names = sorted(
        {
            f.get("name")
            for f in faces
            if f.get("known") and f.get("name") not in (None, "", "Unknown")
        }
    )
    unknown_face_count = sum(1 for f in faces if not f.get("known"))

    return {
        "object_counts": dict(sorted(object_counts.items())),
        "known_names": known_names,
        "unknown_face_count": unknown_face_count,
    }


def scene_has_meaningful_change(previous_signature: dict | None, current_signature: dict) -> bool:
    if previous_signature is None:
        return True
    return previous_signature != current_signature


def build_empty_scene_event() -> tuple[str, str, str | None]:
    return (
        "Scene Cleared",
        "Nothing detected in the scene.",
        None,
    )


# ── Workers ───────────────────────────────────────────────────────────────────
def ai_processing_worker():
    global latest_detections, latest_faces
    global recording_active, recording_end_time, recording_frames, recording_meta
    global latest_event_payload, ui_status, last_event_end_time, last_alert_signature

    while True:
        frame = ai_queue.get()
        if frame is None:
            break

        try:
            detections, _ = run_detection(frame)
            has_person = any(d["label"].lower() == "person" for d in detections)
            has_activity = len(detections) > 0
            faces = identify_faces(frame) if has_person else []
            current_signature = build_scene_signature(detections, faces)

            with annotation_lock:
                latest_detections = detections
                latest_faces = faces

            with recording_lock:
                already_recording = recording_active

            can_start_new_event = (time.time() - last_event_end_time) >= EVENT_REARM_SECONDS
            has_new_scene_change = scene_has_meaningful_change(last_alert_signature, current_signature)

            # Scene cleared
            if not has_activity:
                if last_alert_signature is not None and can_start_new_event:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    snapshot_path = os.path.join(ALERT_IMG_DIR, f"snapshot_{timestamp}.jpg")
                    cv2.imwrite(snapshot_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                    title, alert_description, identity_text = build_empty_scene_event()

                    structured_event = {
                        "time": timestamp,
                        "objects": [],
                        "faces": [],
                        "face_names": [],
                        "person_count": 0,
                        "alert_title": title,
                        "alert_description": alert_description,
                        "identity_text": identity_text,
                        "snapshot": snapshot_path,
                    }

                    send_alert(
                        snapshot_path,
                        alert_description,
                        [],
                        title=title,
                        zone_text=None,
                        identity_text=identity_text,
                    )

                    with latest_event_lock:
                        latest_event_payload = {
                            "id": f"evt_{timestamp}",
                            "description": alert_description,
                            "snapshot": snapshot_path,
                            "structured_event": structured_event,
                            "video_path": None,
                        }
                        ui_status = "idle"

                    last_alert_signature = None
                    last_event_end_time = time.time()
                    print("🚨 EVENT:", structured_event)

            # Any meaningful new scene change triggers an immediate new alert.
            if has_activity and can_start_new_event and has_new_scene_change:
                face_names = [f["name"] for f in (faces or [])]

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snapshot_path = os.path.join(ALERT_IMG_DIR, f"snapshot_{timestamp}.jpg")
                cv2.imwrite(snapshot_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                title, alert_description, identity_text = build_alert_message(detections, faces)

                structured_event = {
                    "time": timestamp,
                    "objects": detections,
                    "faces": faces,
                    "face_names": face_names,
                    "person_count": sum(1 for d in detections if d["label"].lower() == "person"),
                    "alert_title": title,
                    "alert_description": alert_description,
                    "identity_text": identity_text,
                    "snapshot": snapshot_path,
                }

                send_alert(
                    snapshot_path,
                    alert_description,
                    detections,
                    title=title,
                    zone_text=None,
                    identity_text=identity_text,
                )

                # Restart the 5-second recording window on every new meaningful event.
                with recording_lock:
                    recording_active = True
                    recording_end_time = time.time() + EVENT_RECORD_SECONDS
                    recording_frames = [frame.copy()]
                    recording_meta = {
                        "timestamp": timestamp,
                        "description": alert_description,
                        "detections": detections,
                        "structured_event": structured_event,
                        "snapshot": snapshot_path,
                    }

                with latest_event_lock:
                    latest_event_payload = {
                        "id": f"evt_{timestamp}",
                        "description": alert_description,
                        "snapshot": snapshot_path,
                        "structured_event": structured_event,
                        "video_path": None,
                    }
                    ui_status = "active"

                last_alert_signature = current_signature
                print("🚨 EVENT:", structured_event)

        except Exception as e:
            print("AI worker error:", e)

        finally:
            ai_queue.task_done()


def camera_capture_worker():
    global global_frame, system_event, system_event_time
    global recording_active, recording_end_time, recording_frames, recording_meta
    global latest_event_payload, ui_status, last_event_end_time
    global latest_detections, latest_faces

    cap = cv2.VideoCapture(0)
    last_ai_time = 0.0

    while True:
        ret, frame_bgr = cap.read()

        if not ret:
            system_event = "Webcam Not Accessible"
            system_event_time = time.time()
            with frame_lock:
                global_frame = create_placeholder_frame("Webcam not accessible")
            time.sleep(1)
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_buffer.append(frame_rgb.copy())

        # Finalize fixed-duration recording
        with recording_lock:
            if recording_active:
                recording_frames.append(frame_rgb.copy())

                if time.time() >= recording_end_time:
                    last_event_end_time = time.time()

                    meta = dict(recording_meta) if recording_meta else {}
                    frames_to_save = list(recording_frames)

                    recording_active = False
                    recording_end_time = 0.0
                    recording_frames = []
                    recording_meta = None

                    timestamp = meta.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
                    description = meta.get("description", "Security event detected")
                    detections_for_log = meta.get("detections", [])
                    structured_event = meta.get("structured_event", {})
                    snapshot_path = meta.get("snapshot")

                    vid_path = save_fixed_event_video(frames_to_save, timestamp)
                    structured_event["video_path"] = vid_path

                    append_event_log(
                        {
                            "timestamp": timestamp,
                            "description": description,
                            "detections": detections_for_log,
                            "video": vid_path,
                            "snapshot": snapshot_path,
                            "structured_event": structured_event,
                        }
                    )

                    with latest_event_lock:
                        latest_event_payload = {
                            "id": f"evt_{timestamp}",
                            "description": description,
                            "snapshot": snapshot_path,
                            "structured_event": structured_event,
                            "video_path": vid_path,
                        }
                        ui_status = "idle"

                    with annotation_lock:
                        latest_detections = []
                        latest_faces = []

                    print(
                        f"✅ EVENT FINISHED: saved {EVENT_RECORD_SECONDS}-second clip -> {vid_path}"
                    )

        # Camera fault detection
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if mean_brightness < 2.0:
            system_event = "Camera Obstructed (Pitch Black)"
            system_event_time = time.time()
        elif lap_var < 1.0 and mean_brightness < 20.0:
            system_event = "Camera Obstructed (Lens Covered)"
            system_event_time = time.time()
        else:
            if system_event and "Obstructed" in system_event:
                system_event = None

        # Continuous AI inference
        now = time.time()
        if not system_event and (now - last_ai_time) >= AI_INFERENCE_INTERVAL:
            if not ai_queue.full():
                try:
                    ai_queue.put_nowait(frame_rgb.copy())
                    last_ai_time = now
                except queue.Full:
                    pass

        # Display frame
        display = frame_bgr.copy()

        with latest_event_lock:
            current_status = ui_status

        if system_event:
            cv2.putText(
                display,
                f"FAULT: {system_event}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
        elif current_status == "active":
            cv2.putText(
                display,
                f"RECORDING EVENT ({EVENT_RECORD_SECONDS}s)",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2,
            )
        else:
            cv2.putText(
                display,
                "Monitoring Live",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        with annotation_lock:
            detections_snap = list(latest_detections)
            faces_snap = list(latest_faces)

        for d in detections_snap:
            if "box" not in d:
                continue
            x1, y1, x2, y2 = d["box"]
            color = (0, 0, 255) if d["label"].lower() == "person" else (226, 43, 138)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                display,
                f"{d['label']} {d['confidence']:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        for face in faces_snap:
            x1, y1, x2, y2 = face["box"]
            color = (0, 200, 0) if face["known"] else (0, 100, 255)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            label = face["name"]
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(display, (x1, y2), (x1 + tw + 4, y2 + th + 8), color, -1)
            cv2.putText(
                display,
                label,
                (x1 + 2, y2 + th + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

        _, buffer = cv2.imencode(".jpg", display)
        with frame_lock:
            global_frame = buffer.tobytes()

        time.sleep(0.01)


# ── Startup ───────────────────────────────────────────────────────────────────
load_known_faces()
threading.Thread(target=camera_capture_worker, daemon=True).start()
threading.Thread(target=ai_processing_worker, daemon=True).start()


# ── Flask routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("setup.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/enroll")
def enroll_page():
    return render_template("enroll.html")


@app.route("/update_camera", methods=["POST"])
def update_camera():
    return redirect("/")


@app.route("/alerts/<filename>")
def serve_alert_video(filename):
    from flask import send_from_directory

    return send_from_directory(ALERT_IMG_DIR, filename)


@app.route("/api/latest_event")
def api_latest_event():
    with latest_event_lock:
        current_event = dict(latest_event_payload) if latest_event_payload else None
        current_status = ui_status

    resp = {
        "system_event": system_event,
        "status": current_status,
        "description": "System Normal" if current_status == "idle" else "Event active",
        "event_id": None,
        "video_path": None,
        "faces": [],
        "structured_event": None,
    }

    if system_event:
        resp["event_id"] = system_event + str(int(system_event_time))
        resp["description"] = system_event

    if current_event:
        resp["description"] = current_event.get("description", resp["description"])
        resp["event_id"] = current_event.get("id")
        resp["video_path"] = current_event.get("video_path")
        resp["structured_event"] = current_event.get("structured_event")

    with annotation_lock:
        resp["faces"] = [{"name": f["name"], "known": f["known"]} for f in latest_faces]

    return jsonify(resp)


# ── Enrollment API ────────────────────────────────────────────────────────────
def _enroll_thread(name: str, target_samples: int = 20, interval: float = 0.15):
    embeddings = []
    best_frame = None
    best_sharpness = -1.0

    with _enroll_lock:
        _enroll_state.update(
            active=True,
            name=name,
            collected=0,
            status="capturing",
            message="Look at the camera…",
        )

    deadline = time.time() + 30

    while time.time() < deadline:
        with _enroll_lock:
            if not _enroll_state["active"]:
                break

        try:
            frame_rgb = list(frame_buffer)[-1].copy()
        except IndexError:
            time.sleep(interval)
            continue

        try:
            faces = _face_app.get(frame_rgb)
        except Exception:
            time.sleep(interval)
            continue

        if len(faces) != 1:
            msg = (
                "No face detected — move closer."
                if len(faces) == 0
                else "Multiple faces — only one person please."
            )
            with _enroll_lock:
                _enroll_state["message"] = msg
            time.sleep(interval)
            continue

        emb = faces[0].normed_embedding
        embeddings.append(emb)

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness > best_sharpness:
            best_sharpness = sharpness
            best_frame = frame_rgb.copy()

        collected = len(embeddings)
        with _enroll_lock:
            _enroll_state["collected"] = collected
            _enroll_state["message"] = f"Captured {collected}/{target_samples} samples…"

        if collected >= target_samples:
            break

        time.sleep(interval)

    if len(embeddings) < max(5, target_samples // 4):
        with _enroll_lock:
            _enroll_state.update(
                active=False,
                status="error",
                message="Not enough samples — try again in better lighting.",
            )
        return

    mean_emb = np.mean(embeddings, axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-6)
    display_name = name.strip().title()
    slug = _slugify(display_name)

    if best_frame is not None:
        cv2.imwrite(
            os.path.join(KNOWN_FACES_DIR, f"{slug}.jpg"),
            cv2.cvtColor(best_frame, cv2.COLOR_RGB2BGR),
        )

    np.savez(
        os.path.join(KNOWN_FACES_DIR, f"{slug}.npz"),
        embedding=mean_emb,
        sample_count=len(embeddings),
    )

    with known_faces_lock:
        global known_faces
        known_faces = [e for e in known_faces if e["name"] != display_name]
        known_faces.append({"name": display_name, "embedding": mean_emb})

    with _enroll_lock:
        _enroll_state.update(
            active=False,
            status="done",
            message=f"✅ '{display_name}' enrolled successfully!",
        )

    print(f"👤 Enrolled: {display_name} ({len(embeddings)} samples)")


@app.route("/api/enroll/start", methods=["POST"])
def api_enroll_start():
    if not FACE_RECOGNITION_AVAILABLE:
        return jsonify({"error": "InsightFace not installed"}), 503

    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()

    if not name:
        return jsonify({"error": "name is required"}), 400

    with _enroll_lock:
        if _enroll_state["active"]:
            return jsonify({"error": "Enrollment already in progress"}), 409

    threading.Thread(target=_enroll_thread, args=(name,), daemon=True).start()
    return jsonify({"status": "started", "name": name})


@app.route("/api/enroll/status")
def api_enroll_status():
    with _enroll_lock:
        return jsonify(dict(_enroll_state))


@app.route("/api/enroll/cancel", methods=["POST"])
def api_enroll_cancel():
    with _enroll_lock:
        _enroll_state.update(active=False, status="idle", message="Cancelled.")
    return jsonify({"status": "cancelled"})


@app.route("/api/enrolled_people")
def api_enrolled_people():
    with known_faces_lock:
        return jsonify({"people": [e["name"] for e in known_faces]})


@app.route("/api/enroll/delete", methods=["POST"])
def api_enroll_delete():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip().title()
    slug = _slugify(name)

    with known_faces_lock:
        global known_faces
        before = len(known_faces)
        known_faces = [e for e in known_faces if e["name"] != name]
        if len(known_faces) == before:
            return jsonify({"error": "Person not found"}), 404

    for ext in (".jpg", ".jpeg", ".png", ".npz"):
        path = os.path.join(KNOWN_FACES_DIR, f"{slug}{ext}")
        if os.path.exists(path):
            os.remove(path)

    return jsonify({"status": "deleted", "name": name})


@app.route("/reload_faces")
def reload_faces():
    load_known_faces()
    with known_faces_lock:
        names = [e["name"] for e in known_faces]
    return jsonify({"status": "ok", "count": len(names), "names": names})


# ── Video feed ────────────────────────────────────────────────────────────────
def generate_feed():
    while True:
        with frame_lock:
            if global_frame is not None:
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + global_frame
                    + b"\r\n"
                )
        time.sleep(1.0 / 30.0)


@app.route("/video_feed")
def video_feed():
    return Response(generate_feed(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    print("\n🚀 Starting HomeWatch AI Pro")
    print("👉 http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False, use_reloader=False)