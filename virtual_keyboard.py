"""
virtual_keyboard.py
-------------------
On-screen "air keyboard" drawn on the camera view.
  * point with the index finger  -> the key under the fingertip is highlighted
  * pinch (thumb + index)        -> that key is typed

"Hover lock": when you pinch, your fingertip moves a little, so the key under
it may change. We therefore type the key that was hovered just BEFORE the
pinch started (the last key seen while the gesture was "point").
"""
import time

import cv2

ROWS = [
    list("1234567890"),
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM"),
    ["SPACE", "BACK", "ENTER"],
]
SPECIAL = {"SPACE": "space", "BACK": "backspace", "ENTER": "enter"}


class VirtualKeyboard:
    def __init__(self, frame_w, frame_h, top_ratio=0.22, bottom_ratio=0.75, width_ratio=0.80):
        # keyboard sits between 22% and 75% of the frame height, so the bottom
        # row (SPACE / BACK / ENTER) is easy to reach without leaving the camera view
        self.keys = []            # list of (label, x1, y1, x2, y2)
        top = int(frame_h * top_ratio)
        bottom = int(frame_h * bottom_ratio)
        rows = len(ROWS)
        key_h = (bottom - top) // rows
        side = int(frame_w * (1 - width_ratio) / 2)   # empty space left and right
        key_w = (frame_w - 2 * side) // 10
        for r, row in enumerate(ROWS):
            y1 = top + r * key_h
            if r == rows - 1:     # bottom row: wide keys
                widths = [5, 2.5, 2.5]
                x = side
                for lab, wf in zip(row, widths):
                    w = int(key_w * wf)
                    self.keys.append((lab, x + 3, y1 + 3, x + w - 3, y1 + key_h - 3))
                    x += w
            else:
                offset = side + (10 - len(row)) * key_w // 2
                for c, lab in enumerate(row):
                    x1 = offset + c * key_w
                    self.keys.append((lab, x1 + 3, y1 + 3, x1 + key_w - 3, y1 + key_h - 3))
        self.hover = None
        self.locked = None        # key hovered just before a pinch
        self.flash_key, self.flash_t = None, 0.0
        self.typed = ""

    def key_at(self, x, y):
        for lab, x1, y1, x2, y2 in self.keys:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return lab
        return None

    def update_hover(self, x, y, is_pointing):
        """Call every frame with the fingertip pixel position."""
        self.hover = self.key_at(x, y)
        if is_pointing:
            self.locked = self.hover

    def press(self):
        """Called on pinch onset. Returns the key name for the controller, or None."""
        lab = self.locked or self.hover
        if lab is None:
            return None
        self.flash_key, self.flash_t = lab, time.time()
        if lab == "BACK":
            self.typed = self.typed[:-1]
        elif lab == "SPACE":
            self.typed += " "
        elif lab == "ENTER":
            self.typed = ""
        else:
            self.typed += lab.lower()
        self.typed = self.typed[-40:]
        return SPECIAL.get(lab, lab)

    def draw(self, frame):
        overlay = frame.copy()
        for lab, x1, y1, x2, y2 in self.keys:
            if lab == self.flash_key and time.time() - self.flash_t < 0.25:
                col = (0, 200, 0)
            elif lab == self.hover:
                col = (0, 180, 255)
            else:
                col = (60, 60, 60)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), col, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        for lab, x1, y1, x2, y2 in self.keys:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (230, 230, 230), 1)
            fs = 0.7 if len(lab) == 1 else 0.55
            (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
            cv2.putText(frame, lab, ((x1 + x2 - tw) // 2, (y1 + y2 + th) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)
        # typed-text box
        top = self.keys[0][2] - 45
        cv2.rectangle(frame, (20, top), (frame.shape[1] - 20, top + 35), (255, 255, 255), -1)
        cv2.putText(frame, self.typed + "|", (28, top + 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 0), 2, cv2.LINE_AA)
