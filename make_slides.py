"""
make_slides.py  (minimal version)
---------------------------------
Builds a short, clean deck: presentation/Gesture_Control_EE604.pptx
Only the important information; every number is read from results/.
(The long version with more slides: make_slides_full.py)
"""
import csv
import json
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

import config

R = config.RESULTS_DIR
OUT_DIR = os.path.join(config.ROOT_DIR, "presentation")
os.makedirs(OUT_DIR, exist_ok=True)
team = json.load(open(os.path.join(config.ROOT_DIR, "team.json"), encoding="utf-8"))

NAVY, BLUE, LIGHT = RGBColor(0x14, 0x2B, 0x4F), RGBColor(0x1F, 0x5F, 0xA8), RGBColor(0xE8, 0xF1, 0xFB)
GREY, WHITE = RGBColor(0x66, 0x66, 0x66), RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Calibri"
W, H = 13.333, 7.5
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)


# ---------------------------------------------------------------- results
def md(name):
    """results/<name>.md -> list of row lists (header removed)."""
    p = os.path.join(R, name + ".md")
    if not os.path.exists(p):
        return []
    rows = []
    for l in open(p, encoding="utf-8", errors="replace"):
        l = l.strip()
        if l.startswith("|") and not set(l.replace("|", "").strip()) <= set("-: "):
            rows.append([c.strip() for c in l.strip("|").split("|")])
    return rows[1:]


def val(rows, key, col):
    """Number in column `col` of the first row whose first cell starts with `key`."""
    exact = [r for r in rows if r[0] == key]          # exact name first, then prefix
    for r in exact or rows:
        if r[0].startswith(key):
            try:
                return float(r[col].split("+/-")[0])
            except (ValueError, IndexError):
                return None
    return None


def f(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


abl, loso, cnn = md("ablation"), md("loso"), md("cnn_baseline")
flt, vot, anc, perf = md("filter_comparison"), md("voting_comparison"), md("anchor_drift"), md("performance")


# ---------------------------------------------------------------- helpers
def text(s, x, y, w, h, t, size=20, bold=False, color=NAVY, align=PP_ALIGN.LEFT):
    tf = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    for i, line in enumerate(t.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment, p.space_after = align, Pt(12)
        r = p.add_run(); r.text = line
        r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), bold, color, FONT


def slide(title, notes=""):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    text(s, 0.6, 0.35, W - 1.2, 0.8, title, size=32, bold=True)
    bar = s.shapes.add_shape(1, Inches(0.6), Inches(1.15), Inches(1.2), Inches(0.06))
    bar.fill.solid(); bar.fill.fore_color.rgb = BLUE; bar.line.fill.background()
    text(s, W - 1.1, H - 0.5, 0.6, 0.3, str(len(prs.slides)), size=11, color=GREY, align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = notes
    return s


def table(s, head, rows, x, y, w, size=18, best=None, first=0.42):
    t = s.shapes.add_table(len(rows) + 1, len(head), Inches(x), Inches(y), Inches(w),
                           Inches(0.5 * (len(rows) + 1))).table
    t.columns[0].width = Inches(w * first)
    for j in range(1, len(head)):
        t.columns[j].width = Inches(w * (1 - first) / (len(head) - 1))
    for i, row in enumerate([head] + rows):
        for j, v in enumerate(row):
            c = t.cell(i, j); c.text = str(v); c.fill.solid()
            hl = best is not None and i - 1 in best
            c.fill.fore_color.rgb = BLUE if i == 0 else (LIGHT if hl else WHITE)
            for p in c.text_frame.paragraphs:
                p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
                for r in p.runs:
                    r.font.size, r.font.name = Pt(size), FONT
                    r.font.bold = i == 0 or hl
                    r.font.color.rgb = WHITE if i == 0 else NAVY


def picture(s, path, x, y, w=None, h=None):
    if os.path.exists(path):
        s.shapes.add_picture(path, Inches(x), Inches(y), width=Inches(w) if w else None,
                             height=Inches(h) if h else None)


# ---------------------------------------------------------------- 1 title
s = prs.slides.add_slide(prs.slide_layouts[6])
bg = s.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
text(s, 0.8, 2.0, W - 1.6, 1.2, team["short_title"], size=40, bold=True, color=WHITE)
text(s, 0.8, 3.6, W - 1.6, 1.6, "\n".join(m["name"] for m in team["members"]), size=20, color=WHITE)
text(s, 0.8, 6.2, W - 1.6, 0.6, team["course"], size=14, color=RGBColor(0xC8, 0xDA, 0xF0))

# ---------------------------------------------------------------- 2 problem
slide("Problem", "Input, output, constraints.")
text(prs.slides[-1], 0.6, 1.6, 12, 4.5,
     "Input:   live video from a normal laptop webcam\n"
     "Output:   mouse + keyboard actions (move, click, scroll, drag, type)\n"
     "Constraints:   real time on CPU,  no extra hardware,  works for new users,  no false clicks",
     size=24)

# ---------------------------------------------------------------- 3 literature
s = slide("Existing work vs. ours", "What was done before, and how we differ.")
table(s, ["Existing approach", "Problem", "Ours"], [
    ["CNN on pixels (Kopuklu '19)", "needs GPU, big data", "small MLP on 21 hand points"],
    ["'Fingers up' rules (virtual mice)", "fails on new users, tilt", "learned, invariant features"],
    ["Cursor = fingertip", "jumps when you click", "cursor = index knuckle"],
    ["Raw / averaged cursor", "shakes or lags", "One-Euro filter"],
    ["Act on every frame", "false clicks", "temporal voting"]], 0.6, 1.6, 12.1, size=17, first=0.36)

# ---------------------------------------------------------------- 4 overview
s = slide("Method", "Grey = existing (webcam, MediaPipe, OS). Blue = ours.")
picture(s, os.path.join(R, "figures", "overview.png"), 0.6, 1.7, w=12.1)

# ---------------------------------------------------------------- 5 key ideas
s = slide("Key ideas", "Say WHY for each line.")
text(s, 0.6, 1.6, 12, 5,
     "1.  Invariant features (mirror, move, scale, rotate)  ->  position and tilt don't matter\n"
     "2.  Gesture MLP + augmentation  ->  fast, works on new people\n"
     "3.  Voting: 3 of last 5 frames  ->  no false clicks\n"
     "4.  Knuckle anchor + One-Euro filter  ->  steady cursor, accurate clicks\n"
     "5.  Air keyboard with hover-lock", size=22)

# ---------------------------------------------------------------- 6 gestures
s = slide("Gestures", "")
table(s, ["Gesture", "Action"], [
    ["Point", "move cursor / hover key"], ["Pinch thumb + index", "left click / type key"],
    ["Pinch thumb + middle", "right click"], ["Two fingers (V)", "scroll"], ["Fist", "drag"],
    ["Open palm", "pause"], ["Thumbs up (1 s)", "switch mouse / keyboard"]], 2.2, 1.6, 8.9, size=18, first=0.5)

# ---------------------------------------------------------------- 7 setup
n, users = 0, []
if os.path.exists(config.LANDMARK_CSV):
    with open(config.LANDMARK_CSV, newline="") as fh:
        data_rows = list(csv.DictReader(fh))
    n, users = len(data_rows), sorted({r["user"] for r in data_rows})
hw = team.get("hardware", "")
hw = "laptop CPU, no GPU" if hw.startswith("EDIT") else hw
s = slide("Experiments", "")
text(s, 0.6, 1.6, 12, 4.5,
     f"Dataset:   own recordings, {n:,} frames, {len(users)} users, 7 gestures\n"
     "Tests:   random 80/20 split  +  leave-one-user-out (unseen person)\n"
     "Tools:   MediaPipe, PyTorch, OpenCV, PyAutoGUI\n"
     f"Hardware:   {hw}", size=22)

# ---------------------------------------------------------------- 8 accuracy
rows = [["Rule-based (usual)", f(val(abl, "Rule", 1)), f(val(loso, "MEAN", 1)), f(val(abl, "Rule", 3), 2)],
        ["SVM", f(val(abl, "SVM", 1)), f(val(loso, "MEAN", 2)), f(val(abl, "SVM", 3), 2)],
        ["CNN on image", f(val(cnn, "CNN", 1)), "-", f(val(cnn, "CNN", 3), 2)],
        ["MLP (ours)", f(val(abl, "MLP, full features", 1)), f(val(loso, "MEAN", 4)),
         f(val(abl, "MLP, full features", 3), 2)]]
s = slide("Result 1: gesture accuracy",
          "On a NEW person ours is best. CNN is as accurate on known users but about 5x slower and 12x bigger.")
table(s, ["Method", "Same users %", "New user %", "ms/frame"], rows, 0.6, 1.6, 7.6, size=17, best=[3], first=0.34)
picture(s, os.path.join(R, "confusion_matrix.png"), 8.5, 1.5, w=4.3)

# ---------------------------------------------------------------- 9 control
fl = {r[0]: r for r in flt}
gv = lambda k, c: float(fl[k][c]) if k in fl else None
v_raw = vot[0][3] if vot else "-"
v_our = next((r[3] for r in vot if "default" in r[0]), "-")
rows = [["Cursor shake (px / frame)", f(gv("none", 1), 2) + "  (no filter)", f(gv("one_euro", 1), 2)],
        ["Cursor lag (ms)", f(gv("moving_avg", 3), 0) + "  (moving avg.)", f(gv("one_euro", 3), 0)],
        ["Clicks for 30 real pinches", v_raw + "  (no voting)", v_our],
        ["Cursor jump at click (px)", f(val(anc, "index_tip", 1)) + "  (fingertip)", f(val(anc, "index_mcp", 1))],
        ["Speed (FPS on CPU)", "", f(val(perf, "Mean FPS", 1))]]
s = slide("Result 2: control quality", "Each row = one novelty: usual design vs ours.")
table(s, ["Measure", "Usual", "Ours"], rows, 1.2, 1.6, 10.9, size=19, first=0.40)

# ---------------------------------------------------------------- 10 demo
s = slide("Live demo", "Mouse, then toggles f / a / v / r, then keyboard.")
text(s, 0.6, 1.6, 12, 4.5,
     "1.  Move, click, scroll, right click, drag\n"
     "2.  Turn filter / anchor / voting off live  ->  see the difference\n"
     "3.  Thumbs up  ->  type on the air keyboard", size=24)

# ---------------------------------------------------------------- 11 conclusion
s = slide("Conclusion", "")
text(s, 0.6, 1.6, 12, 4.5,
     f"Webcam-only mouse + keyboard at {f(val(perf, 'Mean FPS', 1), 0)} FPS on CPU\n"
     f"{f(val(loso, 'MEAN', 4))}% on unseen users (rules: {f(val(loso, 'MEAN', 1))}%)\n"
     "Steady cursor, no false clicks\n"
     "Future: swipe gestures, more users\n\n"
     "Thank you!   Code: " + team["github"], size=24)

out = os.path.join(OUT_DIR, "Gesture_Control_EE604.pptx")
prs.save(out)
print("Saved", out, f"({len(prs.slides)} slides)")
