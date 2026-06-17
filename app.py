import streamlit as st
import cv2
import sys
import os
import time
import pandas as pd
import numpy as np

# Ensure imports resolve relative to this file's directory
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from detection import DrowsinessDetector, LOGS_PATH, EYE_CLOSE_ALARM_SECONDS
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Driver Drowsiness Monitor",
    page_icon="🚗",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Outfit', sans-serif; 
    }
    
    .stApp { 
        background: radial-gradient(circle at 50% 50%, #0d0d12 0%, #030305 100%) !important; 
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(8, 8, 12, 0.75) !important;
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
        color: #00f2fe;
        font-weight: 700;
        text-shadow: 0 0 15px rgba(0, 242, 254, 0.25);
        letter-spacing: -0.5px;
    }

    /* Metric card base styling */
    .metric-card {
        background: rgba(12, 12, 18, 0.65);
        backdrop-filter: blur(16px) saturate(180%);
        -webkit-backdrop-filter: blur(16px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        margin-bottom: 12px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4), inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 242, 254, 0.3);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.6), 0 0 20px rgba(0, 242, 254, 0.05), inset 0 1px 0 0 rgba(255, 255, 255, 0.08);
    }

    /* State-based glowing cards */
    .metric-card-active {
        border-color: rgba(0, 230, 118, 0.25);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4), 0 0 15px rgba(0, 230, 118, 0.05), inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
    }
    
    .metric-card-active:hover {
        border-color: rgba(0, 230, 118, 0.45);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.6), 0 0 25px rgba(0, 230, 118, 0.15), inset 0 1px 0 0 rgba(255, 255, 255, 0.08);
    }

    .metric-card-drowsy {
        border-color: rgba(255, 23, 68, 0.4);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4), 0 0 20px rgba(255, 23, 68, 0.15), inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
        animation: pulse-border 1.5s infinite alternate;
    }

    @keyframes pulse-border {
        0% {
            border-color: rgba(255, 23, 68, 0.3);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4), 0 0 15px rgba(255, 23, 68, 0.1), inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
        }
        100% {
            border-color: rgba(255, 23, 68, 0.85);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4), 0 0 30px rgba(255, 23, 68, 0.35), inset 0 1px 0 0 rgba(255, 255, 255, 0.08);
        }
    }

    .metric-card-safe {
        border-color: rgba(0, 242, 254, 0.1);
    }
    
    .metric-card-safe:hover {
        border-color: rgba(0, 242, 254, 0.35);
    }

    .metric-card-warn {
        border-color: rgba(255, 23, 68, 0.3);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4), 0 0 15px rgba(255, 23, 68, 0.1), inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
    }

    /* Metric internal label details */
    .metric-label { 
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
        font-size: 0.8rem; 
        color: #8f9bb3; 
        text-transform: uppercase; 
        letter-spacing: 1.2px; 
    }
    
    .metric-value { 
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.3rem; 
        font-weight: 700; 
        color: #fff; 
        margin: 6px 0; 
    }
    
    .metric-sub { 
        font-family: 'Outfit', sans-serif;
        font-size: 0.8rem; 
        color: #4f5b72; 
    }

    /* Status classes */
    .status-active { 
        color: #00e676; 
        font-weight: 700; 
        font-size: 1.8rem; 
        text-shadow: 0 0 12px rgba(0,230,118,0.35);
    }
    
    .status-drowsy { 
        color: #ff1744; 
        font-weight: 700; 
        font-size: 1.8rem; 
        text-shadow: 0 0 15px rgba(255,23,68,0.55);
        animation: blink 0.8s step-start infinite; 
    }
    
    @keyframes blink { 50% { opacity: 0.3; } }

    /* Custom headers and separators */
    .section-title { 
        font-family: 'Outfit', sans-serif;
        font-size: 1.2rem; 
        font-weight: 600; 
        color: #8f9bb3; 
        margin: 18px 0 8px 0; 
        letter-spacing: 0.8px; 
        text-transform: uppercase;
    }
    
    hr { 
        border-color: rgba(255,255,255,0.06); 
    }
    
    /* Styling camera and visual feeds */
    [data-testid="stImage"] img, [data-testid="element-container"] img {
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 0 12px 40px rgba(0,0,0,0.6) !important;
    }
    
    /* Custom Title Style */
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(90deg, #ffffff 0%, #c4cbd9 50%, #7d8496 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        font-family: 'Outfit', sans-serif;
        font-weight: 400;
        font-size: 1.05rem;
        color: rgba(143, 155, 179, 0.7);
        margin-bottom: 24px;
        letter-spacing: 0.3px;
    }
</style>
""", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🚗 Driver Drowsiness Monitor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time detection using Eye Aspect Ratio · Mouth Aspect Ratio · Head Tilt</p>', unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    ear_thresh       = st.slider("EAR Threshold (Eye Closure)",  0.10, 0.40, 0.25, 0.01)
    mar_thresh       = st.slider("MAR Threshold (Yawning)",      0.40, 1.00, 0.70, 0.05)
    tilt_thresh      = st.slider("Head Tilt Threshold (°)",      5.0,  35.0, 15.0, 1.0)
    eye_alarm_secs   = st.slider("⏱️ Eye-Close Alarm Delay (s)", 5,    30,   15,   1)

    st.markdown("---")
    st.markdown("### 📌 Legend")
    st.markdown("🟢 **Green** = Normal  \n🔴 **Red** = Alert")
    st.markdown("---")
    st.markdown("### ℹ️ Thresholds")
    st.markdown(f"- EAR < **{ear_thresh:.2f}** → Eyes closed")
    st.markdown(f"- MAR > **{mar_thresh:.2f}** → Yawning")
    st.markdown(f"- Tilt > **{tilt_thresh:.0f}°** → Head drop")
    st.markdown(f"- Eyes closed ≥ **{eye_alarm_secs}s** → 🔔 Alarm")

# ── Layout ────────────────────────────────────────────────────────────────────
col_cam, col_info = st.columns([2, 1])

with col_cam:
    st.markdown('<p class="section-title">📷 Live Camera Feed</p>', unsafe_allow_html=True)
    cam_placeholder = st.empty()

with col_info:
    st.markdown('<p class="section-title">📊 Live Metrics</p>', unsafe_allow_html=True)
    status_placeholder   = st.empty()
    ear_placeholder      = st.empty()
    mar_placeholder      = st.empty()
    tilt_placeholder     = st.empty()
    countdown_placeholder = st.empty()

st.markdown("---")
st.markdown('<p class="section-title">📋 Event Log (Last 10 Drowsy Events)</p>', unsafe_allow_html=True)
log_placeholder = st.empty()

# ── Helpers ───────────────────────────────────────────────────────────────────
def render_metrics(status, ear, mar, tilt, ear_t, mar_t, tilt_t, eye_closed_secs, alarm_secs):
    # Status card classes
    status_card_class = "metric-card-drowsy" if status == "DROWSY" else "metric-card-active"
    status_class = "status-drowsy" if status == "DROWSY" else "status-active"
    status_icon  = "😴 DROWSY" if status == "DROWSY" else "🟢 ACTIVE"
    status_placeholder.markdown(
        f'<div class="metric-card {status_card_class}">'
        f'<div class="metric-label">Status</div>'
        f'<div class="{status_class}">{status_icon}</div>'
        f'</div>', 
        unsafe_allow_html=True
    )

    # Determine metric safety states
    ear_is_safe = ear >= ear_t
    mar_is_safe = mar <= mar_t
    tilt_is_safe = tilt <= tilt_t

    ear_card_class = "metric-card-safe" if ear_is_safe else "metric-card-warn"
    mar_card_class = "metric-card-safe" if mar_is_safe else "metric-card-warn"
    tilt_card_class = "metric-card-safe" if tilt_is_safe else "metric-card-warn"

    ear_val_color = "#00f2fe" if ear_is_safe else "#ff1744"
    mar_val_color = "#00f2fe" if mar_is_safe else "#ff1744"
    tilt_val_color = "#00f2fe" if tilt_is_safe else "#ff1744"

    ear_placeholder.markdown(
        f'<div class="metric-card {ear_card_class}">'
        f'<div class="metric-label">EAR · Eye Aspect Ratio</div>'
        f'<div class="metric-value" style="color:{ear_val_color}">{ear:.3f}</div>'
        f'<div class="metric-sub">Threshold: {ear_t:.2f}</div>'
        f'</div>', 
        unsafe_allow_html=True
    )
    
    mar_placeholder.markdown(
        f'<div class="metric-card {mar_card_class}">'
        f'<div class="metric-label">MAR · Mouth Aspect Ratio</div>'
        f'<div class="metric-value" style="color:{mar_val_color}">{mar:.3f}</div>'
        f'<div class="metric-sub">Threshold: {mar_t:.2f}</div>'
        f'</div>', 
        unsafe_allow_html=True
    )
    
    tilt_placeholder.markdown(
        f'<div class="metric-card {tilt_card_class}">'
        f'<div class="metric-label">Head Tilt Angle</div>'
        f'<div class="metric-value" style="color:{tilt_val_color}">{tilt:.1f}°</div>'
        f'<div class="metric-sub">Threshold: {tilt_t:.0f}°</div>'
        f'</div>', 
        unsafe_allow_html=True
    )

    # Eye-closed countdown bar
    if eye_closed_secs > 0:
        pct = min(eye_closed_secs / alarm_secs, 1.0)
        # Smooth color transition from yellow to orange to red
        bar_color = "#ff1744" if pct >= 0.75 else ("#ff9100" if pct >= 0.40 else "#00f2fe")
        remaining = max(alarm_secs - eye_closed_secs, 0)
        shadow_style = f"box-shadow: 0 0 10px {bar_color};"
        countdown_placeholder.markdown(
            f'<div class="metric-card metric-card-warn">'
            f'<div class="metric-label">⏱️ Eyes Closed — Alarm in {remaining:.1f}s</div>'
            f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.05);border-radius:8px;height:14px;margin:10px 0;overflow:hidden;">'
            f'<div style="width:{pct*100:.1f}%;background:{bar_color};height:14px;border-radius:6px;transition:width 0.1s ease-out;{shadow_style}"></div>'
            f'</div>'
            f'<div class="metric-sub">{eye_closed_secs:.1f}s / {alarm_secs}s</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        countdown_placeholder.empty()

def render_logs():
    try:
        if os.path.exists(LOGS_PATH):
            df = pd.read_csv(LOGS_PATH)
            if not df.empty:
                drowsy_df = df[df['status'] == 'DROWSY'].tail(10)
                if not drowsy_df.empty:
                    log_placeholder.dataframe(
                        drowsy_df[::-1].reset_index(drop=True),
                        use_container_width=True
                    )
                else:
                    log_placeholder.info("No drowsy events recorded yet.")
            else:
                log_placeholder.info("No events recorded yet.")
    except Exception as e:
        log_placeholder.error(f"Log error: {e}")

# ── Camera loop ───────────────────────────────────────────────────────────────
# ── Input source selection ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📹 Input Source")
    input_source = st.radio(
        "Select video source:",
        ("Live Webcam", "Upload Video"),
        index=0
    )

import tempfile
import os

import detection as _det
_det.EYE_CLOSE_ALARM_SECONDS = eye_alarm_secs   # apply sidebar setting
detector = DrowsinessDetector(ear_thresh, mar_thresh, tilt_thresh)

cap = None
is_webcam = False
temp_path = None

if input_source == "Live Webcam":
    cap = cv2.VideoCapture(0)
    is_webcam = True
    if not cap.isOpened():
        st.error("❌ Cannot open webcam. If you are running on a cloud server like Streamlit Cloud, please select 'Upload Video' in the sidebar.")
        st.stop()
else:
    uploaded_file = st.sidebar.file_uploader("Upload a driving video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        temp_path = tfile.name
        tfile.close()
        cap = cv2.VideoCapture(temp_path)
    else:
        st.info("ℹ️ Please upload a driving video in the sidebar to run the detection demo.")
        st.stop()

cam_placeholder.info("📸 Starting feed…")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            if not is_webcam:
                # Loop video back to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                cam_placeholder.error("❌ Lost connection to webcam.")
                break

        if is_webcam:
            frame = cv2.flip(frame, 1)

        processed_frame, status, ear, mar, tilt = detector.process_frame(frame)
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        # Calculate how long eyes have been continuously closed
        eye_closed_secs = (
            time.time() - detector.eye_closed_since
            if detector.eye_closed_since is not None else 0.0
        )

        cam_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
        render_metrics(status, ear, mar, tilt, ear_thresh, mar_thresh, tilt_thresh,
                       eye_closed_secs, eye_alarm_secs)
        render_logs()

        time.sleep(0.03)  # ~30 fps

except Exception as e:
    st.error(f"Runtime error: {e}")
finally:
    if cap is not None:
        cap.release()
    if temp_path is not None and os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except Exception:
            pass
