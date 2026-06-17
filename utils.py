import numpy as np
from scipy.spatial import distance

def calculate_ear(eye):
    """
    Compute the Eye Aspect Ratio.
    eye is a list of 6 (x, y) coordinates.
    """
    # Vertical distances
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # Horizontal distance
    C = distance.euclidean(eye[0], eye[3])
    # EAR
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_mar(mouth):
    """
    Compute the Mouth Aspect Ratio.
    mouth is a list of 6 (x, y) coordinates representing inner lips.
    Indices: 0: left, 1: top-left, 2: top-right, 3: right, 4: bottom-right, 5: bottom-left
    """
    A = distance.euclidean(mouth[1], mouth[5])
    B = distance.euclidean(mouth[2], mouth[4])
    C = distance.euclidean(mouth[0], mouth[3])
    mar = (A + B) / (2.0 * C)
    return mar

def calculate_head_tilt(nose, chin):
    """
    Calculate head tilt deviation from straight posture.
    Returns deviation in degrees.
    """
    dx = chin[0] - nose[0]
    dy = chin[1] - nose[1]
    
    # Angle relative to positive X-axis (downwards is positive Y in images)
    # If head is straight, chin is directly below nose -> angle should be ~90 degrees
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Calculate deviation from 90 degrees
    deviation = abs(angle - 90.0)
    return deviation
