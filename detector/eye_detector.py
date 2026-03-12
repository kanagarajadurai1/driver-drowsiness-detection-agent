import numpy as np

# ─── MediaPipe Face Mesh Landmark Indices ────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Correct mouth landmarks for new mediapipe Tasks API
# Top lip center, Bottom lip center, Left corner, Right corner
# Using inner lip landmarks for accurate MAR
MOUTH_OUTER = [13, 14, 78, 308, 82, 87, 312, 317]
#               top  bot  left  right  top2 top3 bot2 bot3

# ─── Distance Helper ─────────────────────────────────────────
def euclidean(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# ─── Eye Aspect Ratio ────────────────────────────────────────
def eye_aspect_ratio(eye_points):
    A = euclidean(eye_points[1], eye_points[5])
    B = euclidean(eye_points[2], eye_points[4])
    C = euclidean(eye_points[0], eye_points[3])
    if C == 0:
        return 0.3
    return (A + B) / (2.0 * C)

# ─── Mouth Aspect Ratio (fixed) ──────────────────────────────
def mouth_aspect_ratio(mouth_points):
    """
    MAR = vertical opening / horizontal width
    Uses:
      mouth_points[0] = top lip (landmark 13)
      mouth_points[1] = bottom lip (landmark 14)
      mouth_points[2] = left corner (landmark 78)
      mouth_points[3] = right corner (landmark 308)
    
    Closed mouth: MAR ≈ 0.0 - 0.1
    Open/yawning: MAR ≈ 0.3 - 0.8
    """
    # Vertical: top lip to bottom lip
    vertical = euclidean(mouth_points[0], mouth_points[1])
    # Horizontal: left corner to right corner
    horizontal = euclidean(mouth_points[2], mouth_points[3])
    if horizontal == 0:
        return 0
    mar = vertical / horizontal
    return mar

# ─── Landmark Extractor ──────────────────────────────────────
def get_landmarks(face_landmarks, indices, img_w, img_h):
    """Works with both old FaceMesh (.landmark) and new Tasks API (list)"""
    points = []
    for idx in indices:
        try:
            # New mediapipe Tasks API — face_landmarks is a plain list
            lm = face_landmarks[idx]
        except TypeError:
            # Old API — face_landmarks has .landmark attribute
            lm = face_landmarks.landmark[idx]
        x = int(lm.x * img_w)
        y = int(lm.y * img_h)
        points.append((x, y))
    return points


# ─── Drowsiness Detector State Machine ───────────────────────
class DrowsinessDetector:
    def __init__(self):
        # Detection thresholds
        self.EAR_THRESHOLD   = 0.22   # Eyes closing if EAR below this
        self.MAR_THRESHOLD   = 0.35   # Yawning if MAR above this (was 0.6, now realistic)
        self.EYE_CLOSED_FRAMES = 15   # Frames before beep
        self.YAWN_FRAMES     = 10     # Frames before yawn song triggers

        # Counters
        self.eye_closed_counter = 0
        self.yawn_counter       = 0
        self.total_blinks       = 0
        self.total_yawns        = 0

        # State
        self.current_ear  = 0.0
        self.current_mar  = 0.0
        self.alert_level  = "SAFE"
        self.yawn_active  = False

        # Alert debounce flags
        self.beep_triggered        = False
        self.voice_triggered       = False
        self.voice_trigger_time    = 0
        self.yawn_triggered        = False

        # Session log
        self.session_events = []

    def update(self, ear, mar, timestamp):
        self.current_ear = round(ear, 3)
        self.current_mar = round(mar, 3)

        alerts = {
            "beep": False,
            "voice_alert": False,
            "yawn_song": False,
            "yawn_stop": False,
            "level": "SAFE",
            "eye_closed_frames": 0,
            "yawn_frames": 0,
        }

        # ── Eye detection ──
        if ear < self.EAR_THRESHOLD:
            self.eye_closed_counter += 1
        else:
            if self.eye_closed_counter >= self.EYE_CLOSED_FRAMES:
                self.total_blinks += 1
            self.eye_closed_counter = 0
            self.beep_triggered     = False
            self.voice_triggered    = False

        if self.eye_closed_counter >= self.EYE_CLOSED_FRAMES:
            if self.eye_closed_counter < 45:
                alerts["level"] = "WARNING"
                if not self.beep_triggered:
                    alerts["beep"] = True
                    self.beep_triggered = True
                    self.session_events.append({"type": "beep", "time": timestamp})
            elif self.eye_closed_counter < 90:
                alerts["level"] = "DANGER"
                if not self.voice_triggered:
                    alerts["voice_alert"]     = True
                    self.voice_triggered      = True
                    self.voice_trigger_time   = timestamp
                    self.session_events.append({"type": "voice_alert", "time": timestamp})
            else:
                alerts["level"] = "CRITICAL"
                if timestamp - self.voice_trigger_time > 5:
                    alerts["voice_alert"]   = True
                    self.voice_trigger_time = timestamp
        else:
            alerts["level"] = "SAFE"

        # ── Yawn detection ──
        if mar > self.MAR_THRESHOLD:
            self.yawn_counter += 1
            if self.yawn_counter >= self.YAWN_FRAMES and not self.yawn_triggered:
                alerts["yawn_song"]   = True
                self.yawn_triggered   = True
                self.yawn_active      = True
                self.total_yawns     += 1
                self.session_events.append({"type": "yawn", "time": timestamp})
        else:
            if self.yawn_active and self.yawn_counter > 0:
                alerts["yawn_stop"] = True
                self.yawn_active    = False
            self.yawn_counter   = 0
            self.yawn_triggered = False

        if self.yawn_active and alerts["level"] == "SAFE":
            alerts["level"] = "WARNING"

        alerts["eye_closed_frames"] = self.eye_closed_counter
        alerts["yawn_frames"]       = self.yawn_counter
        self.alert_level            = alerts["level"]
        return alerts

    def get_stats(self):
        return {
            "ear":              self.current_ear,
            "mar":              self.current_mar,
            "alert_level":      self.alert_level,
            "blinks":           self.total_blinks,
            "yawns":            self.total_yawns,
            "eye_closed_frames":self.eye_closed_counter,
            "yawn_frames":      self.yawn_counter,
            "drowsiness_score": self.calculate_drowsiness_score(),
        }

    def calculate_drowsiness_score(self):
        eye_score  = min(100, (self.eye_closed_counter / 90) * 100)
        yawn_score = min(100, (self.yawn_counter / 30) * 50)
        return round(max(eye_score, yawn_score))

    def reset(self):
        self.eye_closed_counter = 0
        self.yawn_counter       = 0
        self.total_blinks       = 0
        self.total_yawns        = 0
        self.beep_triggered     = False
        self.voice_triggered    = False
        self.yawn_triggered     = False
        self.yawn_active        = False
        self.session_events     = []