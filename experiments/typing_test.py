"""
experiments/typing_test.py
--------------------------
Typing-speed test for the air keyboard.

    python experiments/typing_test.py --method air_keyboard --user guna

1. Run app.py in another window and switch to KEYBOARD mode (thumbs-up, hold 1 s).
2. Click once inside the text box of this test window (so it receives the keys).
3. Type the phrase shown, then "type" ENTER.
The timer starts at the first character. Results are appended to
results/typing_summary.md : words per minute (WPM = chars/5 per minute) and
character error rate (edit distance / phrase length).
"""
import argparse
import os
import random
import sys
import time
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

PHRASES = ["hello world", "image processing", "the quick brown fox", "iit kanpur",
           "gesture control", "deep learning", "open the door", "good morning"]


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[len(b)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="air_keyboard")
    ap.add_argument("--user", required=True)
    args = ap.parse_args()
    phrase = random.choice(PHRASES)

    root = tk.Tk(); root.title("Typing test")
    tk.Label(root, text="Type this phrase, then press ENTER:", font=("Arial", 14)).pack(pady=8)
    tk.Label(root, text=phrase, font=("Consolas", 24, "bold")).pack(pady=8)
    entry = tk.Entry(root, font=("Consolas", 22), width=30); entry.pack(pady=8); entry.focus_set()
    result = tk.Label(root, text="", font=("Arial", 14)); result.pack(pady=8)
    state = {"t0": None}

    def on_key(e):
        if state["t0"] is None and e.keysym not in ("Return", "BackSpace"):
            state["t0"] = time.perf_counter()

    def on_enter(e):
        if state["t0"] is None:
            return
        dt = time.perf_counter() - state["t0"]
        typed = entry.get().strip().lower()
        wpm = (len(typed) / 5) / (dt / 60)
        cer = 100 * edit_distance(typed, phrase) / len(phrase)
        result.config(text=f"time {dt:.1f} s   WPM {wpm:.2f}   char error {cer:.1f}%")
        summ = os.path.join(config.RESULTS_DIR, "typing_summary.md")
        new = not os.path.exists(summ)
        with open(summ, "a") as f:
            if new:
                f.write("| Method | User | Phrase | Time (s) | WPM | Char error (%) |\n|---|---|---|---|---|---|\n")
            f.write(f"| {args.method} | {args.user} | {phrase} | {dt:.1f} | {wpm:.2f} | {cer:.1f} |\n")
        print(result.cget("text"))

    entry.bind("<Key>", on_key)
    entry.bind("<Return>", on_enter)
    root.mainloop()


if __name__ == "__main__":
    main()
