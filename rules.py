"""
rules.py
--------
Hand-written rule-based gesture classifier ("which fingers are up?").
This is the classic approach used by most open-source gesture-mouse projects.
We use it as a BASELINE to compare against our learned classifier,
and as a fallback so the app still runs before any model is trained.
"""
import numpy as np

import config
from features import normalize_landmarks

PINCH_THRESHOLD = 0.35   # fingertip distance / palm size


def fingers_up(p):
    """p: normalized landmarks (palm pointing up, -y is up). Returns [thumb, index, middle, ring, pinky]."""
    up = []
    # thumb: tip far from the index knuckle (sideways)
    up.append(np.linalg.norm(p[4, :2] - p[5, :2]) > 0.55)
    # other fingers: tip farther from wrist than the PIP joint
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        up.append(np.linalg.norm(p[tip, :2]) > np.linalg.norm(p[pip, :2]) * 1.05)
    return up


def classify_rules(landmarks, handedness="Right"):
    """Returns (label_index, confidence). Confidence is always 1.0 for rules."""
    p = normalize_landmarks(landmarks, handedness, rotate=True)
    thumb, index, middle, ring, pinky = fingers_up(p)
    d_ti = np.linalg.norm(p[4] - p[8])
    d_tm = np.linalg.norm(p[4] - p[12])
    G = config.GESTURES.index

    if d_ti < PINCH_THRESHOLD and middle:
        return G("pinch_index"), 1.0
    if d_tm < PINCH_THRESHOLD and (ring or pinky):   # ring/pinky up, so not a resting thumb
        return G("pinch_middle"), 1.0
    if index and middle and not ring and not pinky:
        return G("two_finger"), 1.0
    if index and not middle and not ring and not pinky:
        return G("point"), 1.0
    if not index and not middle and not ring and not pinky:
        # thumb up and thumb tip is the highest point of the hand (in raw image)
        raw = np.asarray(landmarks)
        if thumb and raw[4, 1] < raw[[8, 12, 16, 20], 1].min():
            return G("thumbs_up"), 1.0
        return G("fist"), 1.0
    return G("idle_palm"), 1.0
