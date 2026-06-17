import numpy as np
from scipy.io import wavfile
import os

def create_alarm(filename="alarm.wav", duration=1.0, freq=1000):
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Generate a square wave beep
    wave = np.sin(2 * np.pi * freq * t)
    wave = np.sign(wave) # Make it harsh/square wave
    wave = wave * 0.5    # Reduce volume
    
    # Add a pulsating effect
    envelope = np.sin(2 * np.pi * 5 * t) > 0
    wave = wave * envelope
    
    audio = (wave * 32767).astype(np.int16)
    wavfile.write(filename, sample_rate, audio)
    print(f"Created {filename}")

if __name__ == "__main__":
    if not os.path.exists("alarm.wav"):
        create_alarm()
    else:
        print("alarm.wav already exists.")
