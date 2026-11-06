"""
experiments/click_test.py
-------------------------
Target-clicking test (a simple Fitts' law task) to measure how usable the
gesture mouse is, compared with a normal mouse / touchpad.

    python experiments/click_test.py --method mouse   --user guna
    python experiments/click_test.py --method gesture --user guna   (run app.py at the same time)

A red circle appears at a random place. Click it as fast as you can.
After every click (hit or miss) a new circle appears.
Results: results/click_test_<method>_<user>.csv and a summary line in
results/click_test_summary.md
  * movement time (s), error rate (%), throughput (bits/s) = ID / time,
    where ID = log2(distance / width + 1)  (Fitts' index of difficulty)
"""
import argparse
import csv
import math
import os
import random
import sys
import time
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


class ClickTest:
    def __init__(self, root, args):
        self.root, self.args = root, args
        self.W, self.H = 1100, 700
        self.c = tk.Canvas(root, width=self.W, height=self.H, bg="white")
        self.c.pack()
        self.c.bind("<Button-1>", self.on_click)
        self.info = self.c.create_text(self.W // 2, 20, text="", font=("Arial", 14))
        self.trial, self.results = 0, []
        self.prev = (self.W // 2, self.H // 2)
        self.next_target()

    def next_target(self):
        r = random.choice(self.args.radii)
        self.r = r
        self.x = random.randint(r + 10, self.W - r - 10)
        self.y = random.randint(r + 40, self.H - r - 10)
        self.c.delete("target")
        self.c.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill="red", tags="target")
        self.t0 = time.perf_counter()
        self.c.itemconfig(self.info, text=f"{self.args.method}: trial {self.trial + 1}/{self.args.trials}")

    def on_click(self, e):
        mt = time.perf_counter() - self.t0
        hit = math.hypot(e.x - self.x, e.y - self.y) <= self.r
        dist = math.hypot(self.x - self.prev[0], self.y - self.prev[1])
        idx = math.log2(dist / (2 * self.r) + 1)
        self.results.append({"trial": self.trial, "radius": self.r, "distance": round(dist, 1),
                             "time_s": round(mt, 3), "hit": int(hit), "ID_bits": round(idx, 3)})
        self.prev = (e.x, e.y)
        self.trial += 1
        if self.trial >= self.args.trials:
            self.finish()
        else:
            self.next_target()

    def finish(self):
        R = self.results[1:]             # first trial ignored (start position unknown)
        hits = [r for r in R if r["hit"]]
        mt = sum(r["time_s"] for r in R) / len(R)
        err = 100 * (1 - len(hits) / len(R))
        tp = sum(r["ID_bits"] / r["time_s"] for r in hits) / max(len(hits), 1)
        tag = f"{self.args.method}_{self.args.user}"
        path = os.path.join(config.RESULTS_DIR, f"click_test_{tag}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(R[0].keys())); w.writeheader(); w.writerows(self.results)
        summ = os.path.join(config.RESULTS_DIR, "click_test_summary.md")
        new = not os.path.exists(summ)
        with open(summ, "a") as f:
            if new:
                f.write("| Method | User | Mean time per target (s) | Error rate (%) | Throughput (bits/s) |\n"
                        "|---|---|---|---|---|\n")
            f.write(f"| {self.args.method} | {self.args.user} | {mt:.2f} | {err:.1f} | {tp:.2f} |\n")
        msg = f"Done!  time {mt:.2f} s   error {err:.1f}%   throughput {tp:.2f} bits/s"
        print(msg)
        self.c.delete("target")
        self.c.itemconfig(self.info, text=msg + "   (close the window)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, help="e.g. mouse, touchpad, gesture, gesture_noFilter")
    ap.add_argument("--user", required=True)
    ap.add_argument("--trials", type=int, default=21)
    ap.add_argument("--radii", type=int, nargs="+", default=[20, 35, 50])
    args = ap.parse_args()
    root = tk.Tk(); root.title("Click test")
    ClickTest(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
