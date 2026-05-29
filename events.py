import time
import uuid
from datetime import datetime

class EventState:
    IDLE = "idle"
    ACTIVE = "active"
    COOLDOWN = "cooldown"

class EventManager:
    def __init__(self, person_stability_threshold=3, event_exit_timeout=10, cooldown_period=30):
        self.state = EventState.IDLE
        self.current_event = None
        self.last_event_end_time = 0
        
        # Config
        self.person_stability_threshold = person_stability_threshold
        self.event_exit_timeout = event_exit_timeout
        self.cooldown_period = cooldown_period
        
        # Internal Counters
        self.consecutive_person_frames = 0
        self.last_seen_person_time = 0

    def process_frame(self, person_detected: bool, snapshot_path: str = None):
        """
        Process a single detection frame and update the state machine.
        Returns multiple flags to the main app so it knows when to trigger notifications:
        (event_started_now, event_ended_now)
        """
        now = time.time()
        event_started_now = False
        event_ended_now = False
        
        # 1. COOLDOWN STATE
        if self.state == EventState.COOLDOWN:
            if now - self.last_event_end_time > self.cooldown_period:
                self.state = EventState.IDLE
            return False, False

        # 2. IDLE STATE
        if self.state == EventState.IDLE:
            if person_detected:
                self.consecutive_person_frames += 1
                if self.consecutive_person_frames >= self.person_stability_threshold:
                    # Transition IDLE -> ACTIVE
                    self.state = EventState.ACTIVE
                    self.current_event = {
                        "id": f"evt_{str(uuid.uuid4())[:8]}",
                        "type": "person_detected",
                        "start_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
                        "end_time": None,
                        "frames_count": self.consecutive_person_frames,
                        "snapshot": snapshot_path,
                        "notified": False
                    }
                    self.last_seen_person_time = now
                    event_started_now = True
            else:
                # Reset counter if detection flickers
                self.consecutive_person_frames = 0
                
            return event_started_now, event_ended_now

        # 3. ACTIVE STATE
        if self.state == EventState.ACTIVE:
            if person_detected:
                self.last_seen_person_time = now
                self.current_event["frames_count"] += 1
            else:
                # Check for exit timeout
                if now - self.last_seen_person_time > self.event_exit_timeout:
                    # Transition ACTIVE -> COOLDOWN
                    self.state = EventState.COOLDOWN
                    self.last_event_end_time = now
                    self.current_event["end_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    self.consecutive_person_frames = 0
                    event_ended_now = True

            return False, event_ended_now

        return False, False

    def mark_notified(self):
        """Called by the main app once the notification is sent for the current event."""
        if self.current_event:
            self.current_event["notified"] = True

    def get_completed_event(self):
        """Helper to fetch the event details after it has ended."""
        if self.state == EventState.COOLDOWN and self.current_event and self.current_event.get("end_time"):
            evt = self.current_event.copy()
            # Clear it out so we don't return it twice
            self.current_event = None
            return evt
        return None
    
    def get_current_event(self):
        return self.current_event
