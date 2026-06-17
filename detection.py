import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, FaceLandmarker, RunningMode
from utils import calculate_ear, calculate_mar, calculate_head_tilt
import pandas as pd
from datetime import datetime
import threading
try:
    import winsound
except ImportError:
    winsound = None
import os

# ── Absolute paths (safe regardless of CWD) ──────────────────────────────────
_DIR       = os.path.dirname(os.path.abspath(__file__))
ALARM_PATH = os.path.join(_DIR, 'alarm.wav')
LOGS_PATH  = os.path.join(_DIR, 'logs.csv')
MODEL_PATH = os.path.join(_DIR, 'face_landmarker.task')

# ── MediaPipe Face Landmarker (Tasks API — mediapipe 0.10+) ───────────────────
base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options   = FaceLandmarkerOptions(
    base_options=base_opts,
    running_mode=RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)
face_landmarker = FaceLandmarker.create_from_options(options)

# ── MediaPipe Face Mesh landmark indices ──────────────────────────────────────
# Left eye  (6 keypoints: outer-corner, top-left, top-right, inner-corner, bot-right, bot-left)
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
# Right eye
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# Inner-lip ring (6 points)
MOUTH     = [78, 82, 312, 308, 317, 87]
# Nose tip & chin
NOSE      = 1
CHIN      = 152

# ── Drawing connections for eyes / mouth ──────────────────────────────────────
DRAW_CONNECTIONS = mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION  # fallback

def _lm_to_px(landmark, img_w, img_h):
    return (int(landmark.x * img_w), int(landmark.y * img_h))

def get_pts(landmarks, indices, img_w, img_h):
    return [_lm_to_px(landmarks[i], img_w, img_h) for i in indices]

# ── Sound ─────────────────────────────────────────────────────────────────────
def play_alarm_sound():
    try:
        if winsound is not None and os.path.exists(ALARM_PATH):
            winsound.PlaySound(ALARM_PATH, winsound.SND_FILENAME)
    except Exception as e:
        print(f"Alarm error: {e}")

# ── Logging ───────────────────────────────────────────────────────────────────
def log_event(status, ear, mar, tilt):
    try:
        row = {
            "time":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "ear":    round(ear,  3),
            "mar":    round(mar,  3),
            "tilt":   round(tilt, 2),
        }
        pd.DataFrame([row]).to_csv(LOGS_PATH, mode='a', header=False, index=False)
    except Exception as e:
        print(f"Logging error: {e}")

# ── Main detector class ───────────────────────────────────────────────────────
# How many seconds of continuous eye closure before the alarm fires
EYE_CLOSE_ALARM_SECONDS = 15

class DrowsinessDetector:
    def __init__(self, ear_thresh=0.25, mar_thresh=0.70, tilt_thresh=15.0):
        self.ear_thresh   = ear_thresh
        self.mar_thresh   = mar_thresh
        self.tilt_thresh  = tilt_thresh
        self.alarm_playing      = False
        # Time-based eye-closure tracking
        self.eye_closed_since   = None   # timestamp when eyes first closed
        self.alarm_triggered    = False  # flag: alarm fired in this closure episode

    def process_frame(self, frame):
        img_h, img_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = face_landmarker.detect(mp_image)

        status = "ACTIVE"
        ear = mar = tilt = 0.0

        if result.face_landmarks:
            lm = result.face_landmarks[0]   # first face only

            left_eye_pts  = get_pts(lm, LEFT_EYE,  img_w, img_h)
            right_eye_pts = get_pts(lm, RIGHT_EYE, img_w, img_h)
            mouth_pts     = get_pts(lm, MOUTH,     img_w, img_h)
            nose_pt       = _lm_to_px(lm[NOSE], img_w, img_h)
            chin_pt       = _lm_to_px(lm[CHIN], img_w, img_h)

            # ── Compute metrics ───────────────────────────────────────────────
            ear  = (calculate_ear(left_eye_pts) + calculate_ear(right_eye_pts)) / 2.0
            mar  = calculate_mar(mouth_pts)
            tilt = calculate_head_tilt(nose_pt, chin_pt)

            # ── Threshold logic ───────────────────────────────────────────────
            eyes_closed = ear < self.ear_thresh
            is_drowsy   = (eyes_closed or
                           mar > self.mar_thresh or
                           tilt > self.tilt_thresh)

            if eyes_closed:
                if self.eye_closed_since is None:
                    import time as _time
                    self.eye_closed_since = _time.time()   # start the clock
                    self.alarm_triggered  = False

                import time as _time
                elapsed = _time.time() - self.eye_closed_since

                status = "DROWSY"
                # Fire alarm only after EYE_CLOSE_ALARM_SECONDS of continuous closure
                if elapsed >= EYE_CLOSE_ALARM_SECONDS and not self.alarm_triggered:
                    self.alarm_triggered = True
                    self.alarm_playing   = True
                    threading.Thread(target=self._alarm_reset, daemon=True).start()
                    log_event(status, ear, mar, tilt)
            else:
                # Eyes are open — reset the closure timer
                self.eye_closed_since = None
                self.alarm_triggered  = False

                if is_drowsy:   # MAR / tilt still flagged
                    status = "DROWSY"
                else:
                    status = "ACTIVE"

            # ── Draw overlays ─────────────────────────────────────────────────
            eye_color   = (0, 0, 255) if ear  < self.ear_thresh  else (0, 255, 0)
            mouth_color = (0, 0, 255) if mar  > self.mar_thresh   else (0, 255, 255)
            tilt_color  = (0, 0, 255) if tilt > self.tilt_thresh  else (255, 165, 0)

            cv2.polylines(frame, [np.array(left_eye_pts,  np.int32)], True, eye_color,   1)
            cv2.polylines(frame, [np.array(right_eye_pts, np.int32)], True, eye_color,   1)
            cv2.polylines(frame, [np.array(mouth_pts,     np.int32)], True, mouth_color, 1)
            cv2.line(frame, nose_pt, chin_pt, tilt_color, 2)
            cv2.circle(frame, nose_pt, 3, (255, 255, 0), -1)
            cv2.circle(frame, chin_pt, 3, (255, 255, 0), -1)

            # Status banner
            banner = (0, 0, 180) if status == "DROWSY" else (0, 110, 0)
            cv2.rectangle(frame, (0, 0), (img_w, 42), banner, -1)
            icon  = "! DROWSY - WAKE UP!" if status == "DROWSY" else "OK  ACTIVE"
            label = f"  {icon}    EAR:{ear:.2f}  MAR:{mar:.2f}  Tilt:{tilt:.1f}deg"
            cv2.putText(frame, label, (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        return frame, status, ear, mar, tilt

    def _alarm_reset(self):
        play_alarm_sound()
        self.alarm_playing = False
