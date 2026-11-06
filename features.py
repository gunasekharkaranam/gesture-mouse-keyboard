"""
features.py
-----------
Turns the 21 raw hand keypoints into a feature vector for the classifier.

Why not just use raw (x, y) values?  Raw values change when the hand moves
to another part of the image, comes closer to the camera, or tilts.
The gesture is the same, so the features should be the same. We therefore:

  1. Mirror left hands so they look like right hands   (left/right invariance)
  2. Move the wrist to the origin                       (position invariance)
  3. Divide by palm size (wrist -> middle knuckle)      (scale / distance invariance)
  4. Rotate so the palm always points "up"              (in-plane rotation invariance)
  5. Add hand-shape features: joint bend angles and fingertip distances

Feature modes (used in the ablation study in train.py):
  "raw"        : 63 raw values
  "norm"       : steps 1-3        (63 values)
  "norm_rot"   : steps 1-4        (63 values)
  "full"       : steps 1-5        (63 + 15 angles + 10 distances = 88 values)
"""
import numpy as np

FEATURE_MODES = ["raw", "norm", "norm_rot", "full"]

FINGERTIPS = [4, 8, 12, 16, 20]
# (a, b, c) triples: angle at joint b between bones b->a and b->c
JOINT_TRIPLES = [
    (1, 2, 3), (2, 3, 4), (0, 1, 2),             # thumb
    (0, 5, 6), (5, 6, 7), (6, 7, 8),             # index
    (0, 9, 10), (9, 10, 11), (10, 11, 12),       # middle
    (0, 13, 14), (13, 14, 15), (14, 15, 16),     # ring
    (0, 17, 18), (17, 18, 19), (18, 19, 20),     # pinky
]


def normalize_landmarks(lm, handedness="Right", rotate=True):
    """Steps 1-4 above. lm: (21, 3) array. Returns a new (21, 3) array."""
    p = np.asarray(lm, dtype=np.float32).copy()

    # 1. mirror left hand
    if handedness == "Left":
        p[:, 0] = 1.0 - p[:, 0]

    # 2. wrist at origin
    p -= p[0]

    # 3. scale by palm size
    palm = np.linalg.norm(p[9, :2]) + 1e-6
    p /= palm

    # 4. rotate (in x-y plane) so wrist -> middle knuckle points straight up (-y)
    if rotate:
        # (after scaling, |p[9]| in x-y is 1). Rotation matrix [[c,-s],[s,c]]
        # with c=-vy, s=-vx maps (vx, vy) to (0, -1), i.e. "up" in image coords.
        vx, vy = p[9, 0], p[9, 1]
        r = np.hypot(vx, vy) + 1e-6
        c, s = -vy / r, -vx / r
        x, y = p[:, 0].copy(), p[:, 1].copy()
        p[:, 0] = c * x - s * y
        p[:, 1] = s * x + c * y
    return p


def joint_angles(p):
    """15 bend angles (radians) at the finger joints."""
    out = []
    for a, b, c in JOINT_TRIPLES:
        v1, v2 = p[a] - p[b], p[c] - p[b]
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        out.append(np.arccos(np.clip(cosang, -1.0, 1.0)))
    return np.array(out, dtype=np.float32)


def fingertip_distances(p):
    """10 distances between every pair of fingertips (already scale-normalized)."""
    out = []
    for i in range(len(FINGERTIPS)):
        for j in range(i + 1, len(FINGERTIPS)):
            out.append(np.linalg.norm(p[FINGERTIPS[i]] - p[FINGERTIPS[j]]))
    return np.array(out, dtype=np.float32)


def extract_features(lm, handedness="Right", mode="full"):
    """Main entry point. Returns a 1-D float32 feature vector."""
    lm = np.asarray(lm, dtype=np.float32).reshape(21, 3)
    if mode == "raw":
        return lm.flatten()
    if mode == "norm":
        return normalize_landmarks(lm, handedness, rotate=False).flatten()
    p = normalize_landmarks(lm, handedness, rotate=True)
    if mode == "norm_rot":
        return p.flatten()
    if mode == "full":
        return np.concatenate([p.flatten(), joint_angles(p), fingertip_distances(p)])
    raise ValueError(f"Unknown feature mode: {mode}")


def feature_dim(mode):
    return {"raw": 63, "norm": 63, "norm_rot": 63, "full": 88}[mode]


def augment_landmarks(lm, rng):
    """
    Data augmentation on raw keypoints (training only):
    small random rotation, scaling, shift and noise. This makes the model
    robust to how different people hold their hand.
    """
    p = np.asarray(lm, dtype=np.float32).reshape(21, 3).copy()
    center = p[0, :2].copy()
    ang = np.deg2rad(rng.uniform(-20, 20))
    c, s = np.cos(ang), np.sin(ang)
    xy = p[:, :2] - center
    xy = xy @ np.array([[c, -s], [s, c]], dtype=np.float32).T
    xy *= rng.uniform(0.85, 1.15)
    p[:, :2] = xy + center + rng.uniform(-0.05, 0.05, size=2)
    p += rng.normal(0, 0.004, size=p.shape).astype(np.float32)
    return p
