import time
from datetime import datetime

import cv2
import joblib
import numpy as np
import pandas as pd

from arduino_link import connect, send_command       # Step 4
from step3_camera import detect_zones, draw_overlay, CAMERA_INDEX   # Step 3
from timetable import get_expected_level              # Step 2

# ---------- Settings ----------
STABLE_SECONDS = 1.5   # a camera reading must stay the same this long before we act on it

# Press 't' in the camera window to jump between these "pretend" moments.
# None means "use the real clock".
DEMO_MOMENTS = [
    ('Monday', '09:00'),      # Expected level 4
    ('Monday', '11:00'),      # Expected level 3
    ('Wednesday', '11:00'),   # Expected level 2
    ('Monday', '16:00'),      # Expected level 0 (free slot)
    None,                     # real clock
]
# -------------------------------

model = joblib.load('zone_model.pkl')
_decision_cache = {}


def decide(expected, actual):
    """Ask the model what action to take (remembers answers to stay fast)."""
    key = (expected, actual)
    if key not in _decision_cache:
        data = pd.DataFrame([[expected, actual]],
                            columns=['Expected_Zone_Level', 'Actual_Zone_Level'])
        _decision_cache[key] = model.predict(data)[0]
    return _decision_cache[key]


def zone_commands(action, occupied):
    """Turn the model's decision into one command per zone.
    - Nobody seen anywhere: all zones follow the model (standby).
    - Otherwise: occupied zones get the action, empty zones are switched OFF."""
    if not any(occupied):
        return [action] * 4
    return [action if occ else 'OFF' for occ in occupied]


class StableReading:
    """Ignores flickers: only accepts a new camera reading after it stays the same for a while."""

    def __init__(self, seconds):
        self.seconds = seconds
        self.stable = (False, False, False, False)
        self.candidate = self.stable
        self.since = time.time()

    def update(self, reading):
        reading = tuple(reading)
        now = time.time()
        if reading != self.candidate:
            self.candidate = reading
            self.since = now
        if now - self.since >= self.seconds:
            self.stable = self.candidate
        return self.stable


def get_moment(index):
    """Return (day, time_text, label) for the current demo moment."""
    moment = DEMO_MOMENTS[index]
    if moment is None:
        now = datetime.now()
        return now.strftime('%A'), now.strftime('%H:%M'), 'REAL CLOCK'
    return moment[0], moment[1], 'DEMO TIME'


def add_info_panel(frame, lines):
    """Add a black strip under the camera picture with the system's status."""
    panel = np.zeros((90, frame.shape[1], 3), np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(panel, text, (10, 25 + i * 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
    return np.vstack([frame, panel])


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Could not open the webcam. Try changing CAMERA_INDEX in step3_camera.py.")
        return

    ser = connect()
    print("Camera and Arduino connected.")
    print("Press 't' in the camera window to change the pretend time, 'q' to quit.\n")

    reading = StableReading(STABLE_SECONDS)
    moment_index = 0
    last_sent = [None, None, None, None]
    last_status = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read from the webcam.")
                break
            frame = cv2.flip(frame, 1)

            # --- Camera: which zones are occupied? (smoothed) ---
            raw_occupied, centers, _ = detect_zones(frame)
            occupied = reading.update(raw_occupied)
            actual = sum(occupied)

            # --- Timetable: what level is expected right now? ---
            day, time_text, label = get_moment(moment_index)
            expected, note = get_expected_level(day, time_text)

            # --- Model: what action? Then one command per zone ---
            action = decide(expected, actual)
            commands = zone_commands(action, occupied)

            # --- Arduino: send only the zones whose command changed ---
            for i in range(4):
                if commands[i] != last_sent[i]:
                    reply = send_command(ser, f'Z{i + 1}', commands[i])
                    print(f"  Zone {i + 1} -> {commands[i]:<6} (Arduino: {reply})")
                    last_sent[i] = commands[i]

            status = (day, time_text, expected, actual, action)
            if status != last_status:
                print(f"{day} {time_text} | Expected {expected} | Actual {actual} "
                      f"| Model decision: {action}")
                last_status = status

            # --- Show everything on screen ---
            draw_overlay(frame, list(occupied), centers, actual)
            frame = add_info_panel(frame, [
                f"{label}: {day} {time_text} ({note})",
                f"Expected={expected}  Actual={actual}  ->  MODEL DECISION: {action}",
                "Zones: " + "  ".join(f"Z{i + 1}={commands[i]}" for i in range(4)),
            ])
            cv2.imshow("Smart Classroom - Full System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('t'):
                moment_index = (moment_index + 1) % len(DEMO_MOMENTS)

    finally:
        # Always switch everything off when the program ends
        try:
            send_command(ser, 'ALL', 'OFF')
            ser.close()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()
        print("Everything switched off. Goodbye.")


if __name__ == '__main__':
    main()
