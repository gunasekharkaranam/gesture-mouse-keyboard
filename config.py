"""
config.py
---------
All settings of the project live in one place so that every script
(data collection, training, the main app, experiments) uses the same values.
"""
import os

# ----------------------------------------------------------------------------
# Folders
# ----------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")          # hand crops for the CNN baseline
LANDMARK_CSV = os.path.join(DATA_DIR, "landmarks.csv")  # 21 hand keypoints per sample
MODEL_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
LOG_DIR = os.path.join(ROOT_DIR, "logs")

for _d in (DATA_DIR, IMAGE_DIR, MODEL_DIR, RESULTS_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

# MediaPipe hand landmark model (downloaded automatically on first run)
HAND_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/latest/hand_landmarker.task")

# Trained gesture classifier
GESTURE_MODEL_PATH = os.path.join(MODEL_DIR, "gesture_mlp.pt")

# ----------------------------------------------------------------------------
# Gesture classes  (index = class id, also the key you press while collecting)
# ----------------------------------------------------------------------------
GESTURES = [
    "idle_palm",     # 0 : open palm          -> do nothing (pause / "clutch")
    "point",         # 1 : only index up      -> move cursor / hover key
    "pinch_index",   # 2 : thumb + index      -> left click / type key
    "pinch_middle",  # 3 : thumb + middle     -> right click
    "two_finger",    # 4 : index + middle up  -> scroll mode
    "fist",          # 5 : closed fist        -> drag (hold left button)
    "thumbs_up",     # 6 : thumb up (hold 1s) -> switch Mouse <-> Keyboard mode
]
NUM_CLASSES = len(GESTURES)

# What each gesture does (shown in the UI legend)
GESTURE_ACTIONS = {
    "idle_palm": "Pause",
    "point": "Move / Hover",
    "pinch_index": "Left click / Type",
    "pinch_middle": "Right click",
    "two_finger": "Scroll",
    "fist": "Drag",
    "thumbs_up": "Switch mode (hold)",
}

# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 540

# Part of the camera frame that is mapped to the full screen.
# A smaller box means less arm movement is needed to reach screen corners.
ACTIVE_REGION = (0.20, 0.15, 0.80, 0.70)   # (x_min, y_min, x_max, y_max) in [0,1]

# ----------------------------------------------------------------------------
# Novelty 1: cursor smoothing (One-Euro filter by default)
# ----------------------------------------------------------------------------
DEFAULT_FILTER = "one_euro"        # one of: none, moving_avg, kalman, one_euro
ONE_EURO_MIN_CUTOFF = 1.0          # lower  -> smoother when the hand is slow
ONE_EURO_BETA = 0.02               # higher -> less lag when the hand is fast
ONE_EURO_D_CUTOFF = 1.0
MOVING_AVG_WINDOW = 5
KALMAN_PROCESS_NOISE = 5.0        # pixel^2 units
KALMAN_MEASURE_NOISE = 20.0

# ----------------------------------------------------------------------------
# Novelty 2: stable anchor point for the cursor
# "index_mcp" = knuckle of the index finger (landmark 5). It hardly moves
# when you pinch, so the cursor does not jump at the moment of a click.
# ----------------------------------------------------------------------------
DEFAULT_ANCHOR = "index_mcp"       # one of: index_tip, index_mcp, palm

# ----------------------------------------------------------------------------
# Novelty 3: confidence + temporal voting (fewer false clicks)
# ----------------------------------------------------------------------------
VOTE_WINDOW = 5          # look at the last N frames
VOTE_MIN_COUNT = 3       # a gesture must win at least this many of them
CONF_THRESHOLD = 0.70    # and have mean confidence above this

# ----------------------------------------------------------------------------
# Action timing
# ----------------------------------------------------------------------------
CLICK_COOLDOWN_S = 0.35       # minimum time between two clicks
MODE_SWITCH_HOLD_S = 1.0      # hold thumbs-up this long to switch mode
SCROLL_GAIN = 2500            # scroll amount per unit of vertical hand motion
SCROLL_DEADZONE = 0.004       # ignore tiny vertical hand motions

# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
SEED = 42
EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
AUGMENT_COPIES = 4            # extra augmented copies of every training sample
