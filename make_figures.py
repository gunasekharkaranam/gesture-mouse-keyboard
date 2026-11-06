"""
make_figures.py
---------------
Draws the figures for the report and the slides (saved in results/figures/):

  overview.pdf/.png   pipeline of the proposed method (no data needed)
  gestures.png        one real hand crop + its normalized keypoints per gesture
  features.png        raw keypoints -> normalized -> rotated (one real sample)
  anchor.png          why the knuckle anchor is stable during a pinch

The last three use the recorded dataset (data/landmarks.csv, data/images/).
"""
import glob
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import config
from features import normalize_landmarks
from hand_tracker import HAND_CONNECTIONS

FIG_DIR = os.path.join(config.RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

OURS = "#d9ecff"       # colour of our blocks
OURS_EDGE = "#1f5fa8"
EXIST = "#eeeeee"      # colour of existing blocks
EXIST_EDGE = "#777777"


# ----------------------------------------------------------------------
# 1. Overview figure
# ----------------------------------------------------------------------
def box(ax, x, y, w, h, title, body, ours=True, fs=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=1.2",
                                fc=OURS if ours else EXIST, ec=OURS_EDGE if ours else EXIST_EDGE, lw=1.2))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=fs + 0.8, weight="bold")
    ax.text(x + w / 2, y + h - 6.3, body, ha="center", va="top", fontsize=fs, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, dashed=False):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#333333",
                                linestyle="--" if dashed else "-", shrinkA=0, shrinkB=0))


def overview():
    fig, ax = plt.subplots(figsize=(7.2, 2.55))
    ax.set_xlim(0, 144); ax.set_ylim(0, 51); ax.axis("off")
    H, Y = 21, 22
    # webcam icon-like box
    box(ax, 0, Y, 15, H, "Webcam", "RGB frame\n960x540\n~30 fps", ours=False)
    box(ax, 18, Y, 19, H, "MediaPipe", "hand landmarker\n21 keypoints\n(x, y, z)", ours=False)
    box(ax, 40, Y, 21, H, "Invariant\nfeatures", "\nmirror, translate,\nscale, rotate\n+15 angles +10 dists", fs=6.6)
    box(ax, 64, Y, 18, H, "Gesture\nMLP", "\n88-128-64-7\n7 gestures\n+ confidence", fs=6.6)
    box(ax, 85, Y, 17, H, "Temporal\nvoting", "\nN=5 frames,\nk=3 votes,\nconf >= 0.7", fs=6.6)
    # action branches
    box(ax, 105, 33, 26, 17, "Mouse mode", "knuckle anchor ->\nOne-Euro filter -> cursor\nclick, scroll, drag", fs=6.4)
    box(ax, 105, 13, 26, 17, "Keyboard mode", "air keyboard\nwith hover-lock\npinch = type key", fs=6.4)
    box(ax, 133.5, Y, 10.5, H, "OS", "\nmouse &\nkeyboard\nevents", ours=False, fs=6.4)
    for x1, x2 in ((15, 18), (37, 40), (61, 64), (82, 85)):
        arrow(ax, x1, Y + H / 2, x2, Y + H / 2)
    arrow(ax, 102, Y + H / 2, 105, 41.5); arrow(ax, 102, Y + H / 2, 105, 21.5)
    arrow(ax, 131, 41.5, 133.5, Y + H / 2 + 3); arrow(ax, 131, 21.5, 133.5, Y + H / 2 - 3)
    ax.text(118, 31.5, "switch mode: thumbs-up, hold 1 s", fontsize=5.4, ha="center", va="center", color="#555555")
    # personalization
    box(ax, 56, 1.5, 34, 15, "Few-shot personalization", "retrain only the last layer\nwith K samples / gesture", fs=6.4)
    arrow(ax, 73, 16.5, 73, Y, dashed=True)
    # legend
    ax.add_patch(FancyBboxPatch((1, 4), 5, 4, boxstyle="round,pad=0.1", fc=EXIST, ec=EXIST_EDGE))
    ax.text(7.5, 6, "existing component", fontsize=6.5, va="center")
    ax.add_patch(FancyBboxPatch((1, 11), 5, 4, boxstyle="round,pad=0.1", fc=OURS, ec=OURS_EDGE))
    ax.text(7.5, 13, "proposed (ours)", fontsize=6.5, va="center")
    fig.tight_layout(pad=0.1)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"overview.{ext}"), dpi=300)
    plt.close(fig)
    print("saved overview")


# ----------------------------------------------------------------------
# Helpers that need data
# ----------------------------------------------------------------------
def load():
    from train import load_dataset
    if not os.path.exists(config.LANDMARK_CSV):
        return None
    return load_dataset()


def medoid_index(lms, hands, idx):
    """Sample closest to the class mean (a 'typical' example)."""
    F = np.stack([normalize_landmarks(lms[i], hands[i]).flatten() for i in idx])
    return idx[np.argmin(np.linalg.norm(F - F.mean(0), axis=1))]


def draw_skel(ax, p, color="#1f5fa8", lw=1.6, ms=10):
    for a, b in HAND_CONNECTIONS:
        ax.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]], color=color, lw=lw)
    ax.scatter(p[:, 0], p[:, 1], s=ms, color="white", edgecolor=color, zorder=3, lw=0.8)


def gestures(data):
    users, labels, hands, lms = data
    n = config.NUM_CLASSES
    fig, axes = plt.subplots(2, n, figsize=(7.2, 2.45))
    for c, g in enumerate(config.GESTURES):
        idx = np.where(labels == c)[0]
        ax_img, ax_sk = axes[0, c], axes[1, c]
        imgs = sorted(glob.glob(os.path.join(config.IMAGE_DIR, g, "*.jpg")))
        if imgs:
            im = cv2.imread(imgs[len(imgs) // 2])
            ax_img.imshow(im[:, :, ::-1])
        ax_img.set_title(g.replace("_", " "), fontsize=7)
        ax_img.axis("off")
        if len(idx):
            p = normalize_landmarks(lms[medoid_index(lms, hands, idx)], hands[idx[0]])
            draw_skel(ax_sk, p)
        ax_sk.set_xlim(-1.6, 1.6); ax_sk.set_ylim(1.0, -2.6); ax_sk.set_aspect("equal"); ax_sk.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(FIG_DIR, "gestures.png"), dpi=300); plt.close(fig)
    print("saved gestures")


def features_fig(data):
    users, labels, hands, lms = data
    idx = np.where(labels == config.GESTURES.index("point"))[0]
    if not len(idx):
        return
    i = idx[len(idx) // 3]
    raw, h = lms[i], hands[i]
    steps = [("(a) raw keypoints (image coords)", raw[:, :2], (0, 1), True),
             ("(b) mirror + translate + scale", normalize_landmarks(raw, h, rotate=False)[:, :2], None, True),
             ("(c) + rotate palm to 'up'", normalize_landmarks(raw, h, rotate=True)[:, :2], None, True)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, (t, p, lim, inv) in zip(axes, steps):
        draw_skel(ax, p)
        ax.set_title(t, fontsize=7.5)
        if lim:
            ax.set_xlim(*lim); ax.set_ylim(1, 0)
        else:
            ax.set_xlim(-1.8, 1.8); ax.set_ylim(1.0, -2.6)
            ax.axhline(0, color="0.8", lw=0.6); ax.axvline(0, color="0.8", lw=0.6)
        ax.set_aspect("equal"); ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG_DIR, "features.png"), dpi=300); plt.close(fig)
    print("saved features")


def anchor_fig(data):
    users, labels, hands, lms = data
    P, Q = config.GESTURES.index("point"), config.GESTURES.index("pinch_index")
    u = users[0]
    ip, iq = np.where((labels == P) & (users == u))[0], np.where((labels == Q) & (users == u))[0]
    if not len(ip) or not len(iq):
        return
    a = normalize_landmarks(lms[medoid_index(lms, hands, ip)], hands[ip[0]])
    b = normalize_landmarks(lms[medoid_index(lms, hands, iq)], hands[iq[0]])
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    draw_skel(ax, a, color="#999999", lw=1.2); draw_skel(ax, b, color="#1f5fa8", lw=1.4)
    for k, name, col in ((8, "fingertip", "#d62728"), (5, "knuckle (ours)", "#2ca02c")):
        ax.annotate("", xy=b[k, :2], xytext=a[k, :2],
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8))
        d = np.linalg.norm(b[k, :2] - a[k, :2])
        ax.text(a[k, 0] + 0.15, a[k, 1], f"{name}\nmoves {d:.2f}", color=col, fontsize=6.5)
    ax.plot([], [], color="#999999", label="point"); ax.plot([], [], color="#1f5fa8", label="pinch")
    ax.legend(fontsize=6.5, loc="lower left")
    ax.set_xlim(-1.8, 2.0); ax.set_ylim(0.6, -2.6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Motion of anchor points from point to pinch\n(units: palm size)", fontsize=7.5)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(FIG_DIR, "anchor.png"), dpi=300); plt.close(fig)
    print("saved anchor")


def main():
    overview()
    data = load()
    if data is None:
        print("No dataset yet: only the overview figure was made.")
        return
    gestures(data)
    features_fig(data)
    anchor_fig(data)


if __name__ == "__main__":
    main()
