# 🚗 Driver Drowsiness Monitor

An elegant, real-time driver drowsiness detection application built using **Streamlit**, **MediaPipe (Face Landmarker Tasks API)**, and **OpenCV**.

This system monitors three critical physiological signals to assess driver state:
1. **Eye Aspect Ratio (EAR)**: Detects eye closure and blinking duration.
2. **Mouth Aspect Ratio (MAR)**: Detects yawning frequency/duration.
3. **Head Tilt Angle**: Detects head drops and nod deviations from standard driving posture.

---

## ✨ Features

- **Real-Time Video Analytics**: Process webcam frames smoothly (up to 30 FPS) with dynamic landmark overlays.
- **Advanced Metric Tracking**:
  - Continuous eye closure monitoring with a visual warning progress bar.
  - Head tilt detection tracking vertical and horizontal posture deviations.
  - Yawning detection using inner lip mapping.
- **Configurable Settings**:
  - Live adjustment of EAR, MAR, and Head Tilt threshold levels via the Streamlit sidebar.
  - Custom Alarm delay duration setting.
- **Audio Alarm Alert**: Triggers a harsh, pulsating square wave alarm (`alarm.wav`) using `winsound` if eyes remain closed beyond the configured duration.
- **History Log**: Keeps a localized history of the last 10 drowsiness events inside `logs.csv` and renders them dynamically in the UI.
- **Premium UI / UX**: Styled with a radial dark gradient theme, glassmorphic interactive cards, state-based glow colors, and smooth hover micro-animations.

---

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.9+
- Windows OS (required for the built-in `winsound` library used for play alarms)

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MohdAhmad08/drowsiy-check.git
   cd drowsiy-check
   ```

2. **Install Dependencies**:
   Ensure you install the required packages:
   ```bash
   pip install streamlit opencv-python mediapipe numpy pandas scipy
   ```

3. **Verify/Generate Audio Asset**:
   If `alarm.wav` is missing, you can run the audio setup script to generate a standard alarm sound:
   ```bash
   python setup_audio.py
   ```

4. **Run the Streamlit Application**:
   ```bash
   streamlit run app.py
   ```

---

## 📂 Project Structure

- `app.py`: Main Streamlit app containing UI layout, styling, camera loop, and metrics rendering.
- `detection.py`: Main `DrowsinessDetector` class containing MediaPipe inference, threshold verification logic, event logging, and alarm triggering.
- `utils.py`: Pure math helper functions for calculating EAR, MAR, and head tilt deviation.
- `setup_audio.py`: Script to generate a harsh square wave audio warning (`alarm.wav`) using NumPy.
- `face_landmarker.task`: Pre-trained MediaPipe Face Landmarker model asset.
- `logs.csv`: Local comma-separated database storing timestamped drowsiness events.

---

## 🛡️ License

This project is open-source and available under the [MIT License](LICENSE).
