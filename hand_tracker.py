"""
hand_tracker.py
---------------
Thin wrapper around the MediaPipe "Hand Landmarker" (Tasks API).

For each camera frame it returns the 21 hand keypoints (x, y, z), whether
it is a left or right hand, and a bounding box. We build on this existing
model (allowed by the course rules); our own work starts after this step.

Landmark numbering (MediaPipe):
    0 wrist
    1-4   thumb  (CMC, MCP, IP, TIP)
    5-8   index  (MCP, PIP, DIP, TIP)
    9-12  middle
    13-16 ring
    17-20 pinky
"""
import os
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import config

# Pairs of landmarks joined by a line when drawing the hand skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]


def ensure_hand_model(path=config.HAND_MODEL_PATH, url=config.HAND_MODEL_URL):
    """Download the MediaPipe hand model the first time (about 7.5 MB)."""
    if not os.path.exists(path):
        print(f"[hand_tracker] Downloading hand model to {path} ...")
        tmp = path + ".part"                     # avoid a broken file if the download stops
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
        print("[hand_tracker] Done.")
    return path


class HandResult:
    """Simple container for one detected hand."""

    def __init__(self, landmarks, handedness, score, frame_shape):
        self.landmarks = landmarks          # (21, 3) array, x,y in [0,1], z relative depth
        self.handedness = handedness        # "Left" or "Right"
        self.score = score                  # detection confidence
        h, w = frame_shape[:2]
        xs, ys = landmarks[:, 0] * w, landmarks[:, 1] * h
        self.bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    def pixel(self, idx, frame_shape):
        """Pixel coordinates of landmark `idx`."""
        h, w = frame_shape[:2]
        return int(self.landmarks[idx, 0] * w), int(self.landmarks[idx, 1] * h)


class HandTracker:
    """Detects at most one hand per frame using MediaPipe in VIDEO mode."""

    def __init__(self, num_hands=1, det_conf=0.5, track_conf=0.5):
        model_path = ensure_hand_model()
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,   # uses tracking between frames -> faster
            num_hands=num_hands,
            min_hand_detection_confidence=det_conf,
            min_hand_presence_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame_bgr):
        """Run detection on one BGR frame. Returns HandResult or None."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode needs strictly increasing timestamps (in ms)
        ts = int(time.monotonic() * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts

        result = self.landmarker.detect_for_video(mp_image, ts)
        if not result.hand_landmarks:
            return None

        lms = result.hand_landmarks[0]
        landmarks = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
        category = result.handedness[0][0]
        return HandResult(landmarks, category.category_name, category.score, frame_bgr.shape)

    def close(self):
        self.landmarker.close()


def draw_hand(frame, hand, color=(0, 255, 0)):
    """Draw the hand skeleton and keypoints on the frame (in place)."""
    pts = [hand.pixel(i, frame.shape) for i in range(21)]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        r = 6 if i in (4, 8, 12, 16, 20) else 4      # fingertips a little bigger
        cv2.circle(frame, p, r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, p, r, color, 1, cv2.LINE_AA)


def crop_hand(frame, hand, size=96, margin=0.25):
    """Square crop around the hand (used to build the image dataset for the CNN baseline)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = hand.bbox
    side = int(max(x2 - x1, y2 - y1) * (1 + 2 * margin))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    x1, y1 = max(cx - side // 2, 0), max(cy - side // 2, 0)
    x2, y2 = min(x1 + side, w), min(y1 + side, h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
