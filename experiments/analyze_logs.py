"""
experiments/analyze_logs.py
---------------------------
Offline analysis of the per-frame logs recorded with  `python app.py --log`.
Because the log keeps the RAW hand positions and RAW per-frame predictions,
we can replay the same session through every filter / voting setting and
compare them fairly on identical input.

    python experiments/analyze_logs.py logs/session_XXXX.csv [more logs ...]

Suggested recording (about 1 minute each, app in --dry-run is fine):
  1. "still"  : point at the screen and hold the hand as still as you can (jitter)
  2. "move"   : move the cursor around in big and small motions           (lag)
  3. "clicks" : point, pinch, point, pinch ... EXACTLY 30 times            (anchor drift, voting)
                 -> in voting_comparison.md the best setting detects close to 30 clicks

Outputs in results/:
  performance.md        FPS and processing time per frame
  filter_comparison.md  jitter (px) and lag (ms) for none / moving_avg / kalman / one_euro
  filter_trace.png      raw vs filtered cursor trace
  voting_comparison.md  label flicker and false gesture onsets with / without voting
  anchor_drift.md       how far the cursor moves at the moment of a click, per anchor point
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config                                     # noqa: E402
from filters import make_filter, FILTERS          # noqa: E402
from voting import GestureVoter                   # noqa: E402
from app import to_screen                         # noqa: E402

SCREEN_W, SCREEN_H = 1920, 1080


def read_log(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def runs_with_hand(rows):
    """Split the log into continuous pieces where a hand was visible."""
    runs, cur = [], []
    for r in rows:
        if r["hand"] == "1":
            cur.append(r)
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    return [r for r in runs if len(r) >= 15]


def gaussian_smooth(x, sigma=2.0):
    """Zero-phase (centered) smoothing -> used as the 'intended' hand path (reference)."""
    k = int(3 * sigma)
    w = np.exp(-0.5 * (np.arange(-k, k + 1) / sigma) ** 2); w /= w.sum()
    xp = np.pad(x, ((k, k), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, d], w, mode="valid") for d in range(x.shape[1])], 1)


# ----------------------------------------------------------------------
def performance(all_rows, out):
    fps = np.array([float(r["fps"]) for r in all_rows if float(r["fps"]) > 0])
    ms = np.array([float(r["proc_ms"]) for r in all_rows])
    with open(out, "w") as f:
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| Frames analysed | {len(all_rows)} |\n")
        f.write(f"| Mean FPS | {fps.mean():.1f} |\n")
        f.write(f"| Processing time per frame, mean (ms) | {ms.mean():.1f} |\n")
        f.write(f"| Processing time per frame, 95th percentile (ms) | {np.percentile(ms, 95):.1f} |\n")
    print(open(out).read())


def filter_comparison(runs, out_md, out_png):
    res = {n: {"jit": [], "err": [], "lag": []} for n in FILTERS}
    trace_done = False
    for run in runs:
        t = np.array([float(r["t"]) for r in run])
        raw = np.array([to_screen(float(r["mcp_x"]), float(r["mcp_y"]), SCREEN_W, SCREEN_H)
                        for r in run])
        ref = gaussian_smooth(raw)
        dtm = np.median(np.diff(t)) if len(t) > 1 else 1 / 30
        speed = np.r_[0, np.linalg.norm(np.diff(ref, axis=0), axis=1) / np.maximum(np.diff(t), 1e-3)]
        still, moving = speed < 150, speed > 400            # px/s

        outs = {}
        for name in FILTERS:
            flt = make_filter(name)
            y = np.array([flt(x[0], x[1], ti) for x, ti in zip(raw, t)])
            outs[name] = y
            step = np.r_[0, np.linalg.norm(np.diff(y, axis=0), axis=1)]
            if still.sum() > 5:
                res[name]["jit"].append(step[still].mean())      # px moved per frame when hand still
            if moving.sum() > 5:
                res[name]["err"].append(np.linalg.norm(y - ref, axis=1)[moving].mean())
                # lag = shift (in frames) that best aligns filtered signal with the reference
                best = min(range(0, 15), key=lambda s: np.mean(np.linalg.norm(
                    y[s:] - ref[:len(ref) - s], axis=1)))
                res[name]["lag"].append(best * dtm * 1000)

        if not trace_done and len(run) > 90:
            n = min(len(run), 240)
            fig, ax = plt.subplots(figsize=(8, 3.2))
            ax.plot(t[:n] - t[0], raw[:n, 0], color="0.75", lw=1, label="raw")
            for name in ("moving_avg", "kalman", "one_euro"):
                ax.plot(t[:n] - t[0], outs[name][:n, 0], lw=1.3, label=name)
            ax.set_xlabel("time (s)"); ax.set_ylabel("cursor x (px)"); ax.legend(ncol=4, fontsize=8)
            fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)
            trace_done = True

    f = lambda v: f"{np.mean(v):.2f}" if v else "n/a"
    with open(out_md, "w") as fo:
        fo.write("| Filter | Jitter when still (px/frame) | Error when moving (px) | Lag (ms) |\n"
                 "|---|---|---|---|\n")
        for name in FILTERS:
            r = res[name]
            fo.write(f"| {name} | {f(r['jit'])} | {f(r['err'])} | {f(r['lag'])} |\n")
    print(open(out_md).read())


def segments(labels):
    """Run-length encoding: list of (label, length)."""
    segs, prev, n = [], None, 0
    for l in labels:
        if l == prev:
            n += 1
        else:
            if prev is not None:
                segs.append((prev, n))
            prev, n = l, 1
    if prev is not None:
        segs.append((prev, n))
    return segs


def voting_comparison(runs, out_md):
    settings = [("No voting (raw per-frame)", None), ("Voting N=3, k=2", (3, 2)),
                ("Voting N=5, k=3 (default)", (5, 3)), ("Voting N=7, k=5", (7, 5))]
    SKIP = 7
    total_t = sum(float(r[-1]["t"]) - float(r[SKIP]["t"]) for r in runs) / 60.0  # minutes
    with open(out_md, "w") as fo:
        fo.write("| Setting | Gesture changes / min | Short false segments (<3 frames) / min | "
                 "Left clicks detected (total) |\n|---|---|---|---|\n")
        for name, s in settings:
            changes = short = clicks = 0
            for run in runs:
                lab = [int(r["raw_label"]) for r in run]
                conf = [float(r["conf"]) for r in run]
                if s is None:
                    out = lab
                else:
                    v = GestureVoter(window=s[0], min_count=s[1])
                    out = [v.update(l, c) for l, c in zip(lab, conf)]
                # skip the first frames of every run: the voter starts from "idle"
                # and needs a few frames to fill its window (same cut for all settings)
                seg = segments(out[SKIP:])
                changes += len(seg) - 1
                short += sum(1 for _, n in seg[1:-1] if n < 3)
                clicks += sum(1 for l, _ in seg if l == config.GESTURES.index("pinch_index"))
            fo.write(f"| {name} | {changes/total_t:.1f} | {short/total_t:.1f} | {clicks} |\n")
    print(open(out_md).read())


def anchor_drift(runs, out_md, before=4):
    """Cursor displacement between `before` frames before a pinch onset and the onset itself."""
    pinch = config.GESTURES.index("pinch_index")
    d = {"index_tip": [], "index_mcp": [], "palm": []}
    cols = {"index_tip": ("tip_x", "tip_y"), "index_mcp": ("mcp_x", "mcp_y"), "palm": ("palm_x", "palm_y")}
    for run in runs:
        st = [int(r["stable_label"]) for r in run]
        for i in range(before, len(run)):
            if st[i] == pinch and st[i - 1] != pinch:
                for a, (cx, cy) in cols.items():
                    p1 = to_screen(float(run[i - before][cx]), float(run[i - before][cy]), SCREEN_W, SCREEN_H)
                    p2 = to_screen(float(run[i][cx]), float(run[i][cy]), SCREEN_W, SCREEN_H)
                    d[a].append(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    with open(out_md, "w") as fo:
        fo.write(f"| Anchor point | Cursor drift during pinch (px), mean | median | clicks |\n|---|---|---|---|\n")
        for a, v in d.items():
            if v:
                fo.write(f"| {a} | {np.mean(v):.1f} | {np.median(v):.1f} | {len(v)} |\n")
            else:
                fo.write(f"| {a} | n/a | n/a | 0 |\n")
    print(open(out_md).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    a = ap.parse_args()
    all_rows, runs = [], []
    for p in a.logs:
        rows = read_log(p)
        all_rows += rows
        runs += runs_with_hand(rows)
    if not runs:
        raise SystemExit("No usable hand segments in the logs.")
    R = config.RESULTS_DIR
    print("== Performance =="); performance(all_rows, os.path.join(R, "performance.md"))
    print("== Filters ==");     filter_comparison(runs, os.path.join(R, "filter_comparison.md"),
                                                  os.path.join(R, "filter_trace.png"))
    print("== Voting ==");      voting_comparison(runs, os.path.join(R, "voting_comparison.md"))
    print("== Anchor drift =="); anchor_drift(runs, os.path.join(R, "anchor_drift.md"))


if __name__ == "__main__":
    main()
