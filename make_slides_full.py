"""
make_slides.py
--------------
Builds the presentation  presentation/Gesture_Control_EE604_full.pptx  from the
results and figures (run make_figures.py and the experiments first; run_all.py
does everything). Every slide has speaker notes: what to say.

Slides follow the course guidelines:
  problem statement -> literature review -> proposed method (steps + why)
  -> experiments (datasets, implementation, tables, figures) -> demo -> conclusion
"""
import json
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

import config

R = config.RESULTS_DIR
FIG = os.path.join(R, "figures")
OUT_DIR = os.path.join(config.ROOT_DIR, "presentation")
os.makedirs(OUT_DIR, exist_ok=True)
team = json.load(open(os.path.join(config.ROOT_DIR, "team.json"), encoding="utf-8"))

NAVY = RGBColor(0x14, 0x2B, 0x4F)
BLUE = RGBColor(0x1F, 0x5F, 0xA8)
LIGHT = RGBColor(0xE8, 0xF1, 0xFB)
GREY = RGBColor(0x55, 0x55, 0x55)
RED = RGBColor(0xAA, 0x00, 0x00)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H = 13.333, 7.5


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def read_md_table(path):
    if not os.path.exists(path):
        return None
    lines = [l.strip() for l in open(path, encoding="utf-8", errors="replace") if l.strip().startswith("|")]
    split = lambda l: [c.strip() for c in l.strip("|").split("|")]
    if len(lines) < 2:
        return None
    head = split(lines[0])
    body = [split(l) for l in lines[1:] if not set(l.replace("|", "").strip()) <= set("-: ")]
    return head, [b for b in body if len(b) == len(head)]


def text(slide, x, y, w, h, s, size=18, bold=False, color=NAVY, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    for i, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run(); r.text = line
        r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), bold, color, FONT
    return tb


def bullets(slide, x, y, w, h, items, size=18):
    """items: list of str or (str, level)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        s, lvl = (it, 0) if isinstance(it, str) else it
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(6)
        r = p.add_run()
        mark = "" if s[:1].isdigit() else ("• " if lvl == 0 else "– ")
        r.text = mark + s
        r.font.size = Pt(size - 2 * lvl); r.font.name = FONT
        r.font.color.rgb = NAVY if lvl == 0 else GREY
        p.level = lvl
    return tb


def slide(title, notes=""):
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(0.95))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
    text(s, 0.45, 0.12, W - 0.9, 0.75, title, size=28, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 0.45, H - 0.42, 9, 0.3, team["short_title"] + "  |  " + team["course"], size=10, color=GREY)
    text(s, W - 1.2, H - 0.42, 0.8, 0.3, str(len(prs.slides)), size=10, color=GREY, align=PP_ALIGN.RIGHT)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def image(s, path, x, y, w=None, h=None, missing_cmd=""):
    if os.path.exists(path):
        kw = {}
        if w: kw["width"] = Inches(w)
        if h: kw["height"] = Inches(h)
        pic = s.shapes.add_picture(path, Inches(x), Inches(y), **kw)
        if w and h is None and pic.height > Inches(H - y - 0.6):       # keep inside slide
            ratio = Inches(H - y - 0.6) / pic.height
            pic.height = int(pic.height * ratio); pic.width = int(pic.width * ratio)
        return pic
    box = s.shapes.add_shape(1, Inches(x), Inches(y), Inches(w or 4), Inches(h or 2.5))
    box.fill.solid(); box.fill.fore_color.rgb = RGBColor(0xF6, 0xF6, 0xF6); box.line.color.rgb = RED
    tf = box.text_frame; tf.text = "Result not available yet:\n" + missing_cmd
    for p in tf.paragraphs:
        for r in p.runs:
            r.font.size, r.font.color.rgb = Pt(14), RED
    return box


def table(s, head, rows, x, y, w, size=13, bold_col=None, highlight=None):
    n_r, n_c = len(rows) + 1, len(head)
    row_h = 0.38
    shp = s.shapes.add_table(n_r, n_c, Inches(x), Inches(y), Inches(w), Inches(row_h * n_r))
    t = shp.table
    if n_c > 1:                                   # wider first column for method names
        first = 0.36 if n_c > 2 else 0.6
        t.columns[0].width = Inches(w * first)
        for j in range(1, n_c):
            t.columns[j].width = Inches(w * (1 - first) / (n_c - 1))
    for j, hd in enumerate(head):
        c = t.cell(0, j); c.text = hd
        c.fill.solid(); c.fill.fore_color.rgb = BLUE
    for i, r in enumerate(rows, 1):
        for j, v in enumerate(r):
            c = t.cell(i, j); c.text = str(v)
            c.fill.solid()
            c.fill.fore_color.rgb = LIGHT if (highlight and highlight(r)) else WHITE
    for i in range(n_r):
        for j in range(n_c):
            for p in t.cell(i, j).text_frame.paragraphs:
                p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
                for run in p.runs:
                    run.font.size = Pt(size); run.font.name = FONT
                    run.font.bold = i == 0 or (highlight is not None and i > 0 and highlight(rows[i - 1]))
                    run.font.color.rgb = WHITE if i == 0 else NAVY
            t.cell(i, j).margin_top = t.cell(i, j).margin_bottom = Inches(0.03)
    return shp


def result_table(s, md, x, y, w, missing_cmd, size=13, keep=None, rename=None, highlight=None):
    t = read_md_table(os.path.join(R, md))
    if not t:
        return image(s, "", x, y, w, 2.0, missing_cmd)
    head, rows = t
    if keep:
        idx = [head.index(k) for k in keep if k in head]
        head = [head[i] for i in idx]; rows = [[r[i] for i in idx] for r in rows]
    if rename:
        head = [rename.get(h, h) for h in head]
    return table(s, head, rows, x, y, w, size=size, highlight=highlight)


# ----------------------------------------------------------------------
# 1. title
# ----------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
text(s, 0.8, 1.6, W - 1.6, 1.6, team["short_title"], size=40, bold=True, color=WHITE)
text(s, 0.8, 3.0, W - 1.6, 1.0, "Robust real-time computer control with only a webcam, "
     "using invariant hand-keypoint learning", size=20, color=RGBColor(0xC8, 0xDA, 0xF0))
text(s, 0.8, 4.4, W - 1.6, 1.4, "\n".join(f"{m['name']}  ({m['roll']})" for m in team["members"]),
     size=18, color=WHITE)
text(s, 0.8, 6.2, W - 1.6, 0.8, team["course"] + "   |   " + team.get("instructor", ""), size=14,
     color=RGBColor(0xC8, 0xDA, 0xF0))
s.notes_slide.notes_text_frame.text = ("Introduce the team. One sentence: we built a system that lets you use "
                                       "the mouse and keyboard with hand gestures in front of a normal webcam.")

# ----------------------------------------------------------------------
# 2. problem statement
# ----------------------------------------------------------------------
s = slide("Problem statement", "Explain input, output and constraints. Stress: no extra hardware, CPU only, "
          "must work for new people and must never click by mistake.")
bullets(s, 0.6, 1.3, 6.0, 5.5, [
    "Input: live RGB video from an ordinary laptop webcam",
    "Output: operating-system mouse and keyboard events",
    ("move, left / right / double click, scroll, drag", 1),
    ("typing on an on-screen \"air keyboard\"", 1),
    "Constraints",
    ("real time (> 20 FPS) on a laptop CPU, no GPU", 1),
    ("no gloves, markers or depth camera", 1),
    ("works for users not seen in training", 1),
    ("steady cursor and no false clicks", 1)], size=20)
bullets(s, 7.0, 1.3, 5.8, 5.5, [
    "Why it matters",
    ("touch-free use: labs, kitchens, hospitals, public screens", 1),
    ("presentations from a distance", 1),
    ("accessibility for people who find small devices hard to use", 1),
    ("every laptop already has a webcam: zero extra cost", 1)], size=20)

# ----------------------------------------------------------------------
# 3. literature review
# ----------------------------------------------------------------------
s = slide("Literature review: what has been done before",
          "Walk through the table. Key message: pixel CNNs are heavy, keypoint detectors are good, "
          "virtual-mouse projects use fixed rules and are never measured.")
table(s, ["Approach", "Examples", "Limitation for our task"], [
    ["Skin colour / contour", "Survey: Rautaray & Agrawal 2015", "fails with lighting, background"],
    ["CNN / 3D-CNN on pixels", "Molchanov CVPR'16, Kopuklu FG'19", "needs GPU and large data"],
    ["Gesture datasets", "Jester (ICCVW'19), HaGRID (WACV'24)", "not our control gestures"],
    ["Hand keypoint detectors", "OpenPose hand (CVPR'17), MediaPipe Hands", "detect hand only, no control"],
    ["Virtual-mouse projects", "MediaPipe + PyAutoGUI (many on GitHub)", "fixed rules, shaky, false clicks"],
    ["Input filtering", "Kalman 1960, One-Euro (CHI'12)", "not combined with gesture control"],
    ["Pointing evaluation", "Fitts 1954, ISO 9241-9 (Soukoreff'04)", "rarely used in gesture projects"]],
    0.6, 1.3, 12.1, size=15)

# ----------------------------------------------------------------------
# 4. how ours is different
# ----------------------------------------------------------------------
s = slide("How our work is different (novelty)",
          "These are the points we must defend. For each, say which problem of the usual design it fixes, "
          "and that we measured it (slides later).")
table(s, ["Usual virtual mouse", "Our system", "Measured by"], [
    ["Fixed 'fingers up' thresholds", "Gesture MLP on invariant keypoint features + augmentation", "accuracy, unseen users"],
    ["Cursor = fingertip (jumps when you pinch)", "Cursor = index knuckle (stable anchor)", "click drift (px)"],
    ["Raw or moving-average cursor", "Speed-adaptive One-Euro filter", "jitter (px), lag (ms)"],
    ["Acts on every frame", "Confidence + temporal voting", "false clicks, flicker"],
    ["Same model for everyone", "Few-shot personalization (last layer only)", "accuracy vs K samples"],
    ["Mouse only", "Air keyboard with hover-lock", "typing speed (WPM)"]],
    0.6, 1.4, 12.1, size=16, highlight=lambda r: False)

# ----------------------------------------------------------------------
# 5. overview figure
# ----------------------------------------------------------------------
s = slide("Proposed method: overview", "Grey = existing parts we build on (webcam, MediaPipe, OS). "
          "Blue = our work. Follow the arrows from left to right.")
image(s, os.path.join(FIG, "overview.png"), 0.4, 1.4, w=12.5, missing_cmd="python make_figures.py")

# ----------------------------------------------------------------------
# 6-10. method steps
# ----------------------------------------------------------------------
s = slide("Step 1-2: hand keypoints and invariant features",
          "MediaPipe gives 21 points. The same gesture gives different raw numbers when the hand moves, comes "
          "closer or tilts. We remove these effects so a small dataset is enough.")
bullets(s, 0.6, 1.3, 5.6, 5.6, [
    "MediaPipe Hand Landmarker: 21 keypoints (x, y, z) per frame",
    "Our normalisation (why: remove nuisance factors)",
    ("mirror left hand -> one model for both hands", 1),
    ("wrist to origin -> position does not matter", 1),
    ("divide by palm size -> distance does not matter", 1),
    ("rotate palm to 'up' -> tilt does not matter", 1),
    "Add hand-shape features: 15 joint angles + 10 fingertip distances",
    "Result: 88 numbers per frame"], size=18)
image(s, os.path.join(FIG, "features.png"), 6.4, 1.5, w=6.6, missing_cmd="python make_figures.py (needs data)")

s = slide("Step 3: Gesture MLP (deep learning)",
          "Why MLP: input is only 88 numbers; tiny network, fast on CPU, ignores background. "
          "Augmentation imitates other people and camera positions.")
bullets(s, 0.6, 1.3, 5.8, 5.6, [
    "Network: 88 -> 128 -> 64 -> 7 (BatchNorm, ReLU, Dropout 0.2)",
    "About 20k parameters, < 1 ms per frame on CPU",
    "Training: AdamW, cosine LR, label smoothing 0.05, 60 epochs",
    "Augmentation: rotation +/-20 deg, scale 0.85-1.15, shift, noise (4 copies)",
    "Why not a CNN on pixels? tested later: slower and bigger",
    "7 gestures: palm, point, pinch-index, pinch-middle, two-finger, fist, thumbs-up"], size=18)
image(s, os.path.join(FIG, "gestures.png"), 6.6, 1.6, w=6.4, missing_cmd="python make_figures.py (needs data)")

s = slide("Step 4: confidence + temporal voting",
          "Between two gestures the classifier is wrong for 1-2 frames; each wrong pinch would be a click. "
          "Voting adds only 2-3 frames of delay.")
bullets(s, 0.6, 1.4, 12, 5.0, [
    "Keep the last N = 5 predictions (class + confidence)",
    "Change the stable gesture only if one class has >= 3 votes AND mean confidence >= 0.7",
    "Actions fire only at the START of a stable gesture, with 0.35 s cooldown between clicks",
    "Why: removes 1-2 frame mistakes that would become false clicks",
    "Cost: about 2-3 frames delay (< 100 ms)"], size=22)

s = slide("Step 5: stable cursor (knuckle anchor + One-Euro filter)",
          "Show the figure: the fingertip moves a lot when you pinch; the knuckle barely moves. "
          "One-Euro: smooth when slow, fast when fast.")
bullets(s, 0.6, 1.3, 6.6, 5.6, [
    "Anchor: index knuckle (landmark 5), not the fingertip",
    ("pinching moves the fingertip -> cursor leaves the target", 1),
    ("the knuckle stays still -> click lands where you pointed", 1),
    "Active area of the frame is mapped to the full screen",
    "One-Euro filter: cutoff = f_min + beta * speed",
    ("slow hand: low cutoff -> no jitter", 1),
    ("fast hand: high cutoff -> no lag", 1),
    "Compared with: no filter, moving average, Kalman"], size=18)
image(s, os.path.join(FIG, "anchor.png"), 7.6, 1.4, w=5.2, missing_cmd="python make_figures.py (needs data)")

s = slide("Step 6: actions, air keyboard and personalization",
          "Explain each gesture's action. Hover-lock: we type the key you pointed at just before the pinch. "
          "Personalization: freeze the network, retrain only the last layer (455 weights) in about a second.")
table(s, ["Gesture", "Mouse mode", "Keyboard mode"], [
    ["open palm", "pause (cursor frozen)", "-"], ["point", "move cursor", "hover key"],
    ["pinch thumb+index", "left / double click", "type key"], ["pinch thumb+middle", "right click", "-"],
    ["two fingers up", "scroll (move up/down)", "-"], ["fist", "drag", "-"],
    ["thumbs up (hold 1 s)", "switch to keyboard", "switch to mouse"]], 0.6, 1.3, 6.6, size=15)
bullets(s, 7.6, 1.3, 5.3, 5.5, [
    "Air keyboard drawn on the camera view",
    "Hover-lock: type the key pointed at just BEFORE the pinch",
    "Few-shot personalization",
    ("freeze the network body", 1),
    ("retrain only the last layer with K samples / gesture", 1),
    ("about 1 second on a CPU", 1)], size=18)

# ----------------------------------------------------------------------
# 11. datasets + implementation
# ----------------------------------------------------------------------
import csv as _csv
if os.path.exists(config.LANDMARK_CSV):
    with open(config.LANDMARK_CSV, newline="") as f:
        rows_ = list(_csv.DictReader(f))
    users_ = sorted({r["user"] for r in rows_})
    ds = f"{len(rows_)} keypoint samples, {len(users_)} users ({', '.join(users_)}), 7 gestures"
else:
    ds = "(record data with collect_data.py)"
s = slide("Experiments: datasets and implementation details",
          "Say why we recorded our own data: no public dataset has our 7 control gestures with keypoints. "
          "Leave-one-user-out = test on a person the model never saw.")
bullets(s, 0.6, 1.3, 6.0, 5.6, [
    "Dataset: our own recordings (collect_data.py)",
    (ds, 1),
    ("keypoints + 96x96 hand crops for every frame", 1),
    ("varied position, distance, tilt", 1),
    "Splits: random 80/20 and leave-one-user-out",
    "Logged live sessions for cursor experiments",
    "User tests: Fitts' click test, typing test"], size=18)
bullets(s, 6.9, 1.3, 6.0, 5.6, [
    "Python, OpenCV, MediaPipe Tasks, PyTorch, scikit-learn, PyAutoGUI",
    "Hardware: " + team.get("hardware", ""),
    "Camera: " + team.get("camera", ""),
    "Latency measured on 1 CPU thread, 1 sample",
    "Code: " + team["github"]], size=18)

# ----------------------------------------------------------------------
# 12-17. results
# ----------------------------------------------------------------------
s = slide("Results: gesture classification and ablation",
          "Read the top and bottom rows: rules vs our MLP. Then show the ablation: raw -> normalised -> rotated -> "
          "full features, and the effect of augmentation.")
result_table(s, "ablation.md", 0.6, 1.3, 7.4, "python train.py", size=13,
             rename={"method": "Method", "acc": "Acc %", "macro_f1": "F1 %", "latency_ms": "ms", "params": "Params"},
             highlight=lambda r: r[0] == "MLP, full features")
image(s, os.path.join(R, "confusion_matrix.png"), 8.3, 1.3, w=4.7, missing_cmd="python train.py")

s = slide("Results: unseen users and image-CNN baseline",
          "Leave-one-user-out is the honest test for a product. Then: the CNN on pixels vs our keypoint MLP.")
text(s, 0.6, 1.2, 6, 0.4, "Leave-one-user-out accuracy (%)", size=16, bold=True)
result_table(s, "loso.md", 0.6, 1.65, 6.2, "python train.py --loso", size=14,
             highlight=lambda r: r[0] == "MEAN")
text(s, 7.1, 1.2, 6, 0.4, "Pixels vs keypoints", size=16, bold=True)
t_cnn = read_md_table(os.path.join(R, "cnn_baseline.md")); t_abl = read_md_table(os.path.join(R, "ablation.md"))
if t_cnn and t_abl:
    h, r = t_cnn; c = dict(zip(h, r[0]))
    m = next((dict(zip(t_abl[0], x)) for x in t_abl[1] if x[0] == "MLP, full features"), None)
    rows = [["CNN on 64x64 image", c["Accuracy (%)"], c["Latency per sample (ms)"], c["Parameters"]]]
    if m:
        rows.append(["MLP on keypoints (ours)", m["acc"], m["latency_ms"], m["params"]])
    table(s, ["Model", "Acc %", "ms", "Params"], rows, 7.1, 1.65, 5.8, size=14,
          highlight=lambda r: "ours" in r[0])
else:
    image(s, "", 7.1, 1.65, 5.8, 2.0, "python train_cnn.py")

s = slide("Results: few-shot personalization",
          "With K samples per gesture from a new user, accuracy goes up. K=0 means no calibration.")
pp = os.path.join(config.ROOT_DIR, "report", "figures", "personalization.png")
image(s, pp, 3.2, 1.3, w=6.9, missing_cmd="python personalize.py --user NAME, then python make_report.py")

s = slide("Results: cursor smoothing",
          "Same recorded hand motion replayed through every filter. One-Euro: low jitter AND low lag.")
result_table(s, "filter_comparison.md", 0.6, 1.3, 6.3, "python experiments/analyze_logs.py logs/*.csv", size=14,
             rename={"Jitter when still (px/frame)": "Jitter px/frame", "Error when moving (px)": "Error px",
                     "Lag (ms)": "Lag ms"}, highlight=lambda r: r[0] == "one_euro")
image(s, os.path.join(R, "filter_trace.png"), 7.1, 1.3, w=5.9, missing_cmd="python experiments/analyze_logs.py")

s = slide("Results: temporal voting and anchor point",
          "Voting: fewer gesture flips and the click count is close to the 30 real clicks. "
          "Anchor: the knuckle moves far less than the fingertip at the moment of a click.")
text(s, 0.6, 1.2, 6, 0.4, "Temporal voting (30 real clicks)", size=16, bold=True)
result_table(s, "voting_comparison.md", 0.6, 1.65, 7.0, "python experiments/analyze_logs.py logs/*.csv", size=12,
             highlight=lambda r: "default" in r[0])
text(s, 8.0, 1.2, 5, 0.4, "Cursor drift during a click (px)", size=16, bold=True)
result_table(s, "anchor_drift.md", 8.0, 1.65, 4.9, "python experiments/analyze_logs.py logs/*.csv", size=13,
             keep=["Anchor point", "Cursor drift during pinch (px), mean", "median"],
             rename={"Cursor drift during pinch (px), mean": "mean"}, highlight=lambda r: r[0] == "index_mcp")

s = slide("Results: speed and user tests",
          "Real time on CPU. Click test vs a normal mouse/touchpad (Fitts' law). Typing speed with the air keyboard.")
text(s, 0.6, 1.2, 4, 0.4, "Speed", size=16, bold=True)
result_table(s, "performance.md", 0.6, 1.65, 4.6, "python experiments/analyze_logs.py logs/*.csv", size=13)
text(s, 5.5, 1.2, 7, 0.4, "Click test (Fitts' law)", size=16, bold=True)
result_table(s, "click_test_summary.md", 5.5, 1.65, 7.4, "python experiments/click_test.py --method gesture --user NAME",
             size=12, rename={"Mean time per target (s)": "Time (s)", "Error rate (%)": "Error %",
                              "Throughput (bits/s)": "TP bits/s"})
text(s, 5.5, 4.6, 7, 0.4, "Typing test", size=16, bold=True)
result_table(s, "typing_summary.md", 5.5, 5.05, 7.4, "python experiments/typing_test.py --user NAME", size=12,
             keep=["Method", "User", "WPM", "Char error (%)"])

# ----------------------------------------------------------------------
# 18. demo, 19. conclusion, 20. thanks
# ----------------------------------------------------------------------
s = slide("Live demo plan", "Follow this order. Keep the app window small in a corner. "
          "Use keys f, a, v, r to switch each novelty off and on live.")
bullets(s, 0.6, 1.3, 12, 5.6, [
    "1. Open a browser: point to move, pinch to open a link, two fingers to scroll a page",
    "2. Right click (thumb + middle) on the desktop; drag a file into a folder with a fist",
    "3. Press 'f' (filter off/on) and 'a' (fingertip vs knuckle): show the shaking and the jump at a click",
    "4. Press 'v' (voting off): show false clicks; on again: stable",
    "5. Press 'r': rule-based vs our MLP with a tilted hand",
    "6. Thumbs-up 1 s: keyboard mode; type 'hello' in Notepad / search box",
    "7. Let the examiner try it (personalized model in 1 second if needed)"], size=20)

s = slide("Conclusion and future work", "Summarise the four novelty points with one number each from the results.")
bullets(s, 0.6, 1.3, 12, 5.6, [
    "A webcam-only mouse + keyboard that runs in real time on a CPU",
    "Learned classifier on invariant keypoints: works for new users and tilted hands",
    "Stable knuckle anchor + One-Euro filter: steady cursor, clicks land on target",
    "Temporal voting: no false clicks; few-shot personalization in about 1 s",
    "Limitations: one hand, absolute mapping, low light, slow typing",
    "Future: dynamic gestures (swipes) with a temporal model, relative cursor mode, larger user study"], size=20)

s = slide("Thank you! Questions?", "")
text(s, 0.6, 2.4, 12, 1.0, "Code: " + team["github"], size=22, align=PP_ALIGN.CENTER)
text(s, 0.6, 3.4, 12, 2.0, "\n".join(f"{m['name']}: {m['contribution']}" for m in team["members"]),
     size=14, color=GREY, align=PP_ALIGN.CENTER)

out = os.path.join(OUT_DIR, "Gesture_Control_EE604_full.pptx")
prs.save(out)
print("Saved", out, f"({len(prs.slides)} slides)")
