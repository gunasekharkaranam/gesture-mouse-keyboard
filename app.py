"""
app.py
------
Main application: control the mouse and keyboard with hand gestures.

    python app.py                       # normal use
    python app.py --dry-run             # show everything but do NOT move the real mouse
    python app.py --log                 # save a per-frame log in logs/ (for experiments)
    python app.py --model models/gesture_mlp_guna.pt   # use a personalized model

Gestures (MOUSE mode)
    point (index up)        move cursor
    pinch thumb+index       left click (pinch twice quickly = double click)
    pinch thumb+middle      right click
    two fingers up          scroll (move hand up / down)
    fist                    drag (hold left button while moving)
    open palm               pause (cursor stays still)
    thumbs up, hold 1 s     switch to KEYBOARD mode (and back)

KEYBOARD mode: point at a key on the air keyboard, pinch to type it.

Keys in the window
    q / ESC quit           c  control on/off      k  switch mode
    f  next filter         a  next anchor point   v  voting on/off
    r  model <-> rule-based classifier            l  logging on/off
"""
import argparse
import csv
import os
import time

import cv2
import numpy as np

import config
from controller import Controller
from filters import make_filter, FILTERS
from hand_tracker import HandTracker, draw_hand
from rules import classify_rules
from virtual_keyboard import VirtualKeyboard
from voting import GestureVoter

ANCHORS = ["index_mcp", "index_tip", "palm"]
G = {name: i for i, name in enumerate(config.GESTURES)}


def anchor_point(lm, anchor):
    """The hand point that drives the cursor (normalized image coords)."""
    if anchor == "index_tip":
        return float(lm[8, 0]), float(lm[8, 1])
    if anchor == "index_mcp":
        return float(lm[5, 0]), float(lm[5, 1])
    p = lm[[0, 5, 17]].mean(0)                      # palm centre
    return float(p[0]), float(p[1])


def to_screen(x, y, screen_w, screen_h, region=config.ACTIVE_REGION):
    """Map a point inside the active region of the camera frame to screen pixels."""
    x0, y0, x1, y1 = region
    u = np.clip((x - x0) / (x1 - x0), 0, 1)
    v = np.clip((y - y0) / (y1 - y0), 0, 1)
    return u * (screen_w - 1), v * (screen_h - 1)


class GestureEngine:
    """
    All the decision logic, separated from the camera so it can be tested.
    step(hand, t) is called once per frame and returns a list of events.
    """

    def __init__(self, controller, predictor=None, filter_name=config.DEFAULT_FILTER,
                 anchor=config.DEFAULT_ANCHOR, use_voting=True, frame_size=(960, 540)):
        self.ctrl = controller
        self.predictor = predictor                # None -> rule-based classifier
        self.use_model = predictor is not None
        self.filter_name = filter_name
        self.filter = make_filter(filter_name)
        self.tip_filter = make_filter("one_euro")  # for keyboard hover
        self.anchor = anchor
        self.voter = GestureVoter(enabled=use_voting)
        self.keyboard = VirtualKeyboard(*frame_size)
        self.frame_w, self.frame_h = frame_size
        self.mode = "MOUSE"
        self.control_on = True
        self.prev_stable = G["idle_palm"]
        self.last_click_t = 0.0
        self.thumbs_start = None
        self.mode_switched = False
        self.scroll_prev_y = None
        self.cursor = (controller.screen_w / 2, controller.screen_h / 2)
        self.raw_label, self.conf, self.stable = G["idle_palm"], 0.0, G["idle_palm"]

    # ------------------------------------------------------------------
    def classify(self, hand):
        if self.use_model and self.predictor is not None:
            return self.predictor.predict(hand.landmarks, hand.handedness)
        return classify_rules(hand.landmarks, hand.handedness)

    def set_filter(self, name):
        self.filter_name, self.filter = name, make_filter(name)

    def _click_allowed(self, t):
        return t - self.last_click_t >= config.CLICK_COOLDOWN_S

    # ------------------------------------------------------------------
    def step(self, hand, t):
        events = []
        if hand is None:                        # hand lost: release everything
            if self.ctrl.mouse_is_down:
                self.ctrl.mouse_up(); events.append("drag_end")
            self.voter.reset(); self.filter.reset(); self.tip_filter.reset()
            self.prev_stable = self.stable = G["idle_palm"]
            self.thumbs_start, self.scroll_prev_y = None, None
            return events

        self.raw_label, self.conf = self.classify(hand)
        s = self.voter.update(self.raw_label, self.conf)
        self.stable = s
        onset = s != self.prev_stable           # gesture just started this frame
        lm = hand.landmarks

        # ----- mode switch: hold thumbs-up -----
        if s == G["thumbs_up"]:
            if self.thumbs_start is None:
                self.thumbs_start, self.mode_switched = t, False
            elif not self.mode_switched and t - self.thumbs_start >= config.MODE_SWITCH_HOLD_S:
                self.mode = "KEYBOARD" if self.mode == "MOUSE" else "MOUSE"
                self.mode_switched = True
                events.append(f"mode_{self.mode.lower()}")
        else:
            self.thumbs_start = None

        if self.mode == "MOUSE":
            events += self._mouse(s, onset, lm, t)
        else:
            events += self._keyboard(s, onset, lm, t)

        self.prev_stable = s
        return events

    # ------------------------------------------------------------------
    def _mouse(self, s, onset, lm, t):
        ev = []
        ax, ay = anchor_point(lm, self.anchor)
        sx, sy = to_screen(ax, ay, self.ctrl.screen_w, self.ctrl.screen_h)
        fx, fy = self.filter(sx, sy, t)          # smooth in screen pixels

        # cursor moves only while pointing or dragging; frozen otherwise
        if s in (G["point"], G["fist"]):
            self.cursor = (fx, fy)
            if self.control_on:
                self.ctrl.move(fx, fy)

        # drag start / end
        if s == G["fist"] and onset:
            if self.control_on: self.ctrl.mouse_down()
            ev.append("drag_start")
        if s != G["fist"] and self.ctrl.mouse_is_down:
            self.ctrl.mouse_up(); ev.append("drag_end")

        # clicks (on gesture onset, with cooldown)
        if onset and s == G["pinch_index"] and self._click_allowed(t):
            if self.control_on: self.ctrl.click("left")
            self.last_click_t = t; ev.append("left_click")
        if onset and s == G["pinch_middle"] and self._click_allowed(t):
            if self.control_on: self.ctrl.click("right")
            self.last_click_t = t; ev.append("right_click")

        # scroll: vertical hand motion while two fingers are up
        if s == G["two_finger"]:
            if self.scroll_prev_y is not None:
                dy = self.scroll_prev_y - ay              # hand up -> positive -> scroll up
                if abs(dy) > config.SCROLL_DEADZONE:
                    amount = int(dy * config.SCROLL_GAIN)
                    if self.control_on: self.ctrl.scroll(amount)
                    ev.append(f"scroll_{amount}")
            self.scroll_prev_y = ay
        else:
            self.scroll_prev_y = None
        return ev

    def _keyboard(self, s, onset, lm, t):
        ev = []
        # fingertip in frame pixels (smoothed) -> hovered key
        px, py = self.tip_filter(lm[8, 0] * self.frame_w, lm[8, 1] * self.frame_h, t)
        self.tip_px = (int(px), int(py))
        self.keyboard.update_hover(px, py, is_pointing=(s == G["point"]))
        if onset and s == G["pinch_index"] and self._click_allowed(t):
            key = self.keyboard.press()
            if key:
                if self.control_on: self.ctrl.press_key(key)
                self.last_click_t = t; ev.append(f"key_{key}")
        return ev


# ----------------------------------------------------------------------
# Drawing the user interface
# ----------------------------------------------------------------------
def draw_ui(frame, eng, hand, fps, logging_on, last_events):
    h, w = frame.shape[:2]
    if eng.mode == "MOUSE":
        x0, y0, x1, y1 = config.ACTIVE_REGION
        cv2.rectangle(frame, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (255, 200, 0), 1)
        cv2.putText(frame, "active area", (int(x0 * w) + 4, int(y0 * h) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
    else:
        eng.keyboard.draw(frame)

    if hand is not None:
        draw_hand(frame, hand)
        if eng.mode == "MOUSE":
            ax, ay = anchor_point(hand.landmarks, eng.anchor)
            cv2.circle(frame, (int(ax * w), int(ay * h)), 9, (0, 0, 255), 2)
        elif hasattr(eng, "tip_px"):
            cv2.circle(frame, eng.tip_px, 10, (0, 0, 255), 2)

    # top status bar
    cv2.rectangle(frame, (0, 0), (w, 64), (25, 25, 25), -1)
    mode_col = (0, 220, 255) if eng.mode == "MOUSE" else (255, 120, 255)
    cv2.putText(frame, eng.mode, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, mode_col, 2)
    gname = config.GESTURES[eng.stable] if hand is not None else "no hand"
    cv2.putText(frame, f"{gname}", (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    # confidence bar
    cv2.rectangle(frame, (150, 40), (350, 52), (90, 90, 90), 1)
    if hand is not None:
        cv2.rectangle(frame, (150, 40), (150 + int(200 * eng.conf), 52), (0, 200, 0), -1)
    info = (f"FPS {fps:4.1f} | clf {'MLP' if eng.use_model else 'rules'} | filter {eng.filter_name}"
            f" | anchor {eng.anchor} | vote {'on' if eng.voter.enabled else 'off'}")
    cv2.putText(frame, info, (370, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    ctl = "CONTROL ON" if eng.control_on else "CONTROL OFF (c)"
    cv2.putText(frame, ctl, (370, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0) if eng.control_on else (0, 0, 255), 1)
    if logging_on:
        cv2.circle(frame, (w - 20, 20), 8, (0, 0, 255), -1)
        cv2.putText(frame, "LOG", (w - 62, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if last_events:
        cv2.putText(frame, last_events, (w - 260, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # mode-switch progress (thumbs-up hold)
    if eng.thumbs_start is not None and not eng.mode_switched:
        p = min((time.perf_counter() - eng.thumbs_start) / config.MODE_SWITCH_HOLD_S, 1.0)
        cv2.ellipse(frame, (w // 2, h // 2), (40, 40), -90, 0, int(360 * p), (255, 120, 255), 6)

    # legend (mouse mode only, to keep the keyboard visible)
    if eng.mode == "MOUSE":
        y = h - 12 - 20 * len(config.GESTURES)
        cv2.rectangle(frame, (0, y - 20), (290, h), (25, 25, 25), -1)
        for i, g in enumerate(config.GESTURES):
            col = (0, 255, 255) if hand is not None and i == eng.stable else (210, 210, 210)
            cv2.putText(frame, f"{g:<13} {config.GESTURE_ACTIONS[g]}", (8, y + 20 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)


# ----------------------------------------------------------------------
# Logging (for the experiments)
# ----------------------------------------------------------------------
LOG_COLS = ["t", "fps", "proc_ms", "hand", "raw_label", "conf", "stable_label",
            "tip_x", "tip_y", "mcp_x", "mcp_y", "palm_x", "palm_y",
            "cursor_x", "cursor_y", "mode", "filter", "anchor", "voting", "events"]


def log_row(eng, hand, t, fps, proc_ms, events):
    """One CSV row per frame: raw keypoints, raw + stable prediction, cursor, events."""
    lm = hand.landmarks if hand is not None else np.full((21, 3), np.nan)
    palm = lm[[0, 5, 17]].mean(0)
    return [f"{t:.4f}", f"{fps:.2f}", f"{proc_ms:.2f}", int(hand is not None),
            eng.raw_label, f"{eng.conf:.3f}", eng.stable,
            f"{lm[8,0]:.5f}", f"{lm[8,1]:.5f}", f"{lm[5,0]:.5f}", f"{lm[5,1]:.5f}",
            f"{palm[0]:.5f}", f"{palm[1]:.5f}",
            f"{eng.cursor[0]:.1f}", f"{eng.cursor[1]:.1f}", eng.mode,
            eng.filter_name, eng.anchor, int(eng.voter.enabled), ";".join(events)]


def open_log():
    path = os.path.join(config.LOG_DIR, time.strftime("session_%Y%m%d_%H%M%S.csv"))
    f = open(path, "w", newline="")
    w = csv.writer(f); w.writerow(LOG_COLS)
    print(f"[log] writing {path}")
    return f, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.GESTURE_MODEL_PATH)
    ap.add_argument("--rules", action="store_true", help="use the rule-based classifier")
    ap.add_argument("--filter", default=config.DEFAULT_FILTER, choices=list(FILTERS))
    ap.add_argument("--anchor", default=config.DEFAULT_ANCHOR, choices=ANCHORS)
    ap.add_argument("--no-vote", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="do not send real mouse/keyboard events")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    args = ap.parse_args()

    predictor = None
    if not args.rules:
        if os.path.exists(args.model):
            from model import GesturePredictor
            predictor = GesturePredictor(args.model)
            print(f"Loaded gesture model {args.model}")
        else:
            print(f"No model at {args.model} -> using rule-based classifier. Run train.py first.")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Camera not found. Try --camera 1")
    fh, fw = frame.shape[:2]

    tracker = HandTracker()
    ctrl = Controller(dry_run=args.dry_run)
    eng = GestureEngine(ctrl, predictor, args.filter, args.anchor, not args.no_vote, (fw, fh))

    win = "Gesture Mouse & Keyboard (EE604)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)   # keep the window above others
    except Exception:
        pass

    log_f, log_w = open_log() if args.log else (None, None)
    fps, t_prev = 0.0, time.perf_counter()
    last_ev, last_ev_t = "", 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        frame = cv2.flip(frame, 1)                 # mirror view
        hand = tracker.process(frame)
        events = eng.step(hand, t0)
        proc_ms = (time.perf_counter() - t0) * 1000

        dt = t0 - t_prev; t_prev = t0
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps
        if events:
            last_ev, last_ev_t = events[-1], t0
        draw_ui(frame, eng, hand, fps, log_w is not None, last_ev if t0 - last_ev_t < 0.8 else "")
        cv2.imshow(win, frame)

        if log_w is not None:
            log_w.writerow(log_row(eng, hand, t0, fps, proc_ms, events))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("c"):
            eng.control_on = not eng.control_on
            ctrl.mouse_up()
        elif key == ord("k"):
            eng.mode = "KEYBOARD" if eng.mode == "MOUSE" else "MOUSE"
        elif key == ord("f"):
            names = list(FILTERS); eng.set_filter(names[(names.index(eng.filter_name) + 1) % len(names)])
        elif key == ord("a"):
            eng.anchor = ANCHORS[(ANCHORS.index(eng.anchor) + 1) % len(ANCHORS)]; eng.filter.reset()
        elif key == ord("v"):
            eng.voter.enabled = not eng.voter.enabled; eng.voter.reset()
        elif key == ord("r"):
            if eng.predictor is not None:
                eng.use_model = not eng.use_model
        elif key == ord("l"):
            if log_w is None:
                log_f, log_w = open_log()
            else:
                log_f.close(); log_f, log_w = None, None

    ctrl.mouse_up()
    if log_f:
        log_f.close()
    cap.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
