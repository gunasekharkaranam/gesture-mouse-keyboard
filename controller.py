"""
controller.py
-------------
Sends the real mouse / keyboard events to the operating system (pyautogui).
With dry_run=True nothing is sent (useful for testing and for logging-only runs).
"""


class Controller:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.mouse_is_down = False
        if dry_run:
            self.screen_w, self.screen_h = 1920, 1080
            return
        import pyautogui
        pyautogui.FAILSAFE = False   # do not abort when the cursor reaches a screen corner
        pyautogui.PAUSE = 0          # no extra delay after each call (we need real time)
        self.pg = pyautogui
        self.screen_w, self.screen_h = pyautogui.size()

    # ---------------- mouse ----------------
    def move(self, x, y):
        x = min(max(int(x), 0), self.screen_w - 1)
        y = min(max(int(y), 0), self.screen_h - 1)
        if not self.dry_run:
            self.pg.moveTo(x, y)

    def click(self, button="left"):
        if not self.dry_run:
            self.pg.click(button=button)

    def mouse_down(self):
        if not self.mouse_is_down and not self.dry_run:
            self.pg.mouseDown()
        self.mouse_is_down = True

    def mouse_up(self):
        if self.mouse_is_down and not self.dry_run:
            self.pg.mouseUp()
        self.mouse_is_down = False

    def scroll(self, amount):
        if amount and not self.dry_run:
            self.pg.scroll(int(amount))

    # ---------------- keyboard ----------------
    def press_key(self, key):
        """key: a single character, or 'space' / 'backspace' / 'enter'."""
        if not self.dry_run:
            self.pg.press(key.lower() if len(key) == 1 else key)
