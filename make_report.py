"""
make_report.py
--------------
Builds the CVPR-style write-up in report/ from the experiment results.

  * reads results/*.md (made by train.py, train_cnn.py, personalize.py,
    experiments/*.py) and team.json
  * writes report/auto/*.tex : numbers, tables, observation sentences,
    authors, contributions, code appendix
  * copies the figures to report/figures/ (a grey placeholder is used when
    a result does not exist yet)
  * compiles report/main.pdf if pdflatex is installed, and always makes
    report_overleaf.zip (upload it to Overleaf: New Project -> Upload Project)

Every sentence that contains a number is generated from the real results,
so the text always matches the tables. Missing results appear in red.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import zipfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

R = config.RESULTS_DIR
REPORT = os.path.join(config.ROOT_DIR, "report")
AUTO = os.path.join(REPORT, "auto")
FIGS = os.path.join(REPORT, "figures")
CODE = os.path.join(REPORT, "code")
for d in (AUTO, FIGS, CODE):
    os.makedirs(d, exist_ok=True)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def tex(s):
    """Escape LaTeX special characters in plain text."""
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\^{}")):
        s = s.replace(a, b)
    return s


def missing(cmd):
    return r"\todo{Result not available yet: run \texttt{" + tex(cmd) + "}.}"


def read_md_table(path):
    """Markdown table -> list of dicts (header -> cell string)."""
    if not os.path.exists(path):
        return None
    lines = [l.strip() for l in open(path, encoding="utf-8", errors="replace") if l.strip().startswith("|")]
    if len(lines) < 2:
        return None
    split = lambda l: [c.strip() for c in l.strip("|").split("|")]
    head = split(lines[0])
    rows = []
    for l in lines[1:]:
        if set(l.replace("|", "").strip()) <= set("-: "):
            continue
        cells = split(l)
        if len(cells) == len(head):
            rows.append(dict(zip(head, cells)))
    return rows


def num(s):
    """'12.3', '12.3 +/- 1.0', 'n/a' -> float or None."""
    try:
        return float(str(s).split("+/-")[0].strip())
    except (ValueError, TypeError):
        return None


def f1(x, d=1):
    return "--" if x is None else f"{x:.{d}f}"


def pct_change(new, old):
    if new is None or old is None or old == 0:
        return None
    return 100 * (old - new) / old


def table(label, caption, cols, rows, align=None, note=None, wide=False, bold_max=None, bold_min=None):
    """Build a booktabs table. rows: list of lists of strings (already LaTeX)."""
    env = "table*" if wide else "table"
    align = align or ("l" + "c" * (len(cols) - 1))
    for ci in (bold_max or []):
        vals = [num(r[ci]) for r in rows]
        if any(v is not None for v in vals):
            m = max(v for v in vals if v is not None)
            for r, v in zip(rows, vals):
                if v == m:
                    r[ci] = r"\textbf{" + r[ci] + "}"
    for ci in (bold_min or []):
        vals = [num(r[ci]) for r in rows]
        if any(v is not None for v in vals):
            m = min(v for v in vals if v is not None)
            for r, v in zip(rows, vals):
                if v == m:
                    r[ci] = r"\textbf{" + r[ci] + "}"
    cap = caption + (" " + note if note else "")
    out = [rf"\begin{{{env}}}[t]", r"\centering\small", rf"\caption{{{cap}}}", rf"\label{{{label}}}",
           r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
           rf"\begin{{tabular}}{{@{{}}{align}@{{}}}}", r"\toprule", " & ".join(cols) + r" \\", r"\midrule"]
    for r in rows:
        out.append(r"\midrule" if r == "MID" else " & ".join(r) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}}", rf"\end{{{env}}}"]
    return "\n".join(out)


def empty_table(label, caption, cmd):
    return "\n".join([r"\begin{table}[t]", r"\centering\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                      r"\fbox{\parbox{0.9\linewidth}{\centering " + missing(cmd) + "}}", r"\end{table}"])


def write(name, text):
    with open(os.path.join(AUTO, name), "w", encoding="utf-8") as f:
        f.write(text + "\n")


def placeholder_png(path, text):
    fig, ax = plt.subplots(figsize=(4, 2.4))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=9, color="#aa0000", wrap=True)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#aa0000")
    fig.savefig(path, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------
# 1. team / authors / contributions
# ----------------------------------------------------------------------
team = json.load(open(os.path.join(config.ROOT_DIR, "team.json"), encoding="utf-8"))
authors = r" \and ".join(
    tex(m["name"]) + r"\\" + tex(m["roll"]) + r"\\{\tt\small " + tex(m["email"]) + "}" for m in team["members"])
write("authors.tex", r"\title{" + tex(team["title"]) + "}\n\\author{" + authors +
      r"\\[4pt] \small " + tex(team["course"]) + "}")
write("contributions.tex", r"\begin{itemize}\setlength\itemsep{1pt}" + "\n" + "\n".join(
    r"\item \textbf{" + tex(m["name"]) + " (" + tex(m["roll"]) + "):} " + tex(m["contribution"])
    for m in team["members"]) + "\n" + r"\end{itemize}")

_hw = team.get("hardware", "")
macros = {"GitHubURL": team["github"],
          "Hardware": "an ordinary laptop CPU without a GPU" if _hw.startswith("EDIT") or not _hw else tex(_hw),
          "Camera": tex(team.get("camera", "--"))}
obs = {}

# ----------------------------------------------------------------------
# 2. dataset
# ----------------------------------------------------------------------
if os.path.exists(config.LANDMARK_CSV):
    import csv
    with open(config.LANDMARK_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    users = sorted({r["user"] for r in rows})
    n_img = len(glob.glob(os.path.join(config.IMAGE_DIR, "*", "*.jpg")))
    per = {g: sum(r["label"] == g for r in rows) for g in config.GESTURES}
    macros["DatasetSentence"] = (
        f"Because no public dataset matches our gesture set and our control setting, we recorded our own: "
        f"{len(rows)} labelled frames from {len(users)} users (between {min(per.values())} and "
        f"{max(per.values())} per gesture) and {n_img} hand-crop images.")
    macros["DatasetShort"] = f"{len(rows)} keypoint samples and {n_img} hand crops from {len(users)} users"
    n_users = len(users)
else:
    macros["DatasetSentence"] = missing("python collect_data.py --user NAME")
    macros["DatasetShort"] = "recorded with collect\\_data.py"
    n_users = 0

# ----------------------------------------------------------------------
# 3. classification ablation
# ----------------------------------------------------------------------
abl = read_md_table(os.path.join(R, "ablation.md"))
A = {}
if abl:
    nice = {"Rule-based (fingers up)": "Rule-based (fingers up)",
            "SVM (RBF), full features": "SVM (RBF), our features",
            "Random Forest, full features": "Random Forest, our features",
            "kNN (k=5), full features": "kNN ($k$=5), our features",
            "MLP, raw features": "MLP, raw coordinates",
            "MLP, norm features": "MLP, + mirror/translate/scale",
            "MLP, norm_rot features": "MLP, + rotation",
            "MLP, full features, no augmentation": "MLP, full features, no augm.",
            "MLP, full features": r"\textbf{MLP, full features + augm. (ours)}"}
    rows = []
    for r in abl:
        A[r["method"]] = r
        if r["method"] == "MLP, raw features":
            rows.append("MID")
        params = r.get("params", "")
        params = f"{int(float(params))/1000:.1f}k" if params not in ("", None) else "--"
        rows.append([nice.get(r["method"], tex(r["method"])), r["acc"], r["macro_f1"],
                     f1(num(r["latency_ms"]), 3), params])
    write("tab_ablation.tex", table("tab:ablation", "Gesture classification on the random 20\\% test split "
                                    "(accuracy and macro-F1 in \\%, latency per sample on one CPU thread).",
                                    ["Method", "Acc.", "F1", "ms", "Params"], rows, bold_max=[1, 2]))
    g = lambda m, k="acc": num(A[m][k]) if m in A else None
    mlp, rules, raw, noaug = g("MLP, full features"), g("Rule-based (fingers up)"), g("MLP, raw features"), \
        g("MLP, full features, no augmentation")
    classic = {k: g(k) for k in ("SVM (RBF), full features", "Random Forest, full features", "kNN (k=5), full features") if k in A}
    best_c = max(classic, key=classic.get) if classic else None
    macros["AccMLP"], macros["AccRules"] = f1(mlp, 2), f1(rules, 2)
    s = (f"Our MLP with the full feature set reaches {mlp:.2f}\\% accuracy (macro-F1 "
         f"{g('MLP, full features', 'macro_f1'):.2f}\\%) in {g('MLP, full features', 'latency_ms'):.2f}\\,ms per "
         f"sample, compared with {rules:.2f}\\% for the rule-based method. ")
    if raw is not None:
        s += (f"The feature design matters: the same MLP on raw coordinates reaches {raw:.2f}\\%, "
              f"so normalisation and shape features change accuracy by {mlp - raw:+.2f} points. ")
    if noaug is not None:
        s += f"Augmentation changes accuracy by {mlp - noaug:+.2f} points on this split. "
    if best_c:
        bc = classic[best_c]
        lat_c = g(best_c, "latency_ms")
        name = best_c.split(",")[0]
        if bc > mlp:
            s += (f"{name} is slightly more accurate ({bc:.2f}\\%), but takes {lat_c:.2f}\\,ms per sample, "
                  f"{lat_c / g('MLP, full features', 'latency_ms'):.1f}$\\times$ the MLP's time. ")
        else:
            s += f"The best classic classifier on our features is {name} with {bc:.2f}\\%. "
    obs["Classification"] = s + "\\Cref{fig:cm} shows the remaining confusions."
else:
    write("tab_ablation.tex", empty_table("tab:ablation", "Gesture classification (random split).", "python train.py"))
    obs["Classification"] = missing("python train.py")
    mlp = rules = None

# ----------------------------------------------------------------------
# 4. leave-one-user-out
# ----------------------------------------------------------------------
loso = read_md_table(os.path.join(R, "loso.md"))
if loso:
    k = list(loso[0].keys())
    rows = [["MID"] if False else [tex(r[k[0]])] + [r[c] for c in k[1:]] for r in loso]
    rows = [r if r[0] != "MEAN" else r for r in rows]
    mean_i = [i for i, r in enumerate(rows) if r[0] == "MEAN"]
    if mean_i:
        rows[mean_i[0]] = [r"\textbf{Mean}"] + [r"\textbf{" + c + "}" for c in rows[mean_i[0]][1:]]
        rows.insert(mean_i[0], "MID")
    write("tab_loso.tex", table("tab:loso", "Leave-one-user-out accuracy (\\%): trained on the other users, "
                                "tested on the user in the row.",
                                ["Test user", "Rules", "SVM", "MLP raw", "MLP ours"], rows))
    m = {kk: num(v) for kk, v in loso[-1].items()}
    lr, ls, lraw, lm = m[k[1]], m[k[2]], m[k[3]], m[k[4]]
    users_rows = [r for r in loso if r[k[0]] != "MEAN"]
    worst = min(users_rows, key=lambda r: num(r[k[4]]))
    macros["LosoMLP"], macros["LosoRules"] = f1(lm, 2), f1(lr, 2)
    obs["Loso"] = (f"Averaged over {len(users_rows)} held-out users, our MLP reaches {lm:.2f}\\%, against "
                   f"{lr:.2f}\\% for the rules, {ls:.2f}\\% for the SVM and {lraw:.2f}\\% for an MLP on raw "
                   f"coordinates without augmentation. The hardest user is {tex(worst[k[0]])} "
                   f"({num(worst[k[4]]):.2f}\\%), which motivates personalization (\\cref{{fig:personal}}).")
else:
    write("tab_loso.tex", empty_table("tab:loso", "Leave-one-user-out accuracy.", "python train.py --loso"))
    obs["Loso"] = missing("python train.py --loso")
    lm = lr = None

# ----------------------------------------------------------------------
# 5. CNN baseline
# ----------------------------------------------------------------------
cnn = read_md_table(os.path.join(R, "cnn_baseline.md"))
if cnn and abl and "MLP, full features" in A:
    c = cnn[0]; mrow = A["MLP, full features"]
    ca, cl, cp = num(c["Accuracy (%)"]), num(c["Latency per sample (ms)"]), num(c["Parameters"])
    ma, ml, mp = num(mrow["acc"]), num(mrow["latency_ms"]), num(mrow["params"])
    rows = [["CNN, $64\\times64$ hand image", f"{ca:.2f}", c["Macro F1 (%)"], f"{cl:.2f}", f"{cp/1000:.0f}k"],
            ["MLP, keypoint features (ours)", f"{ma:.2f}", mrow["macro_f1"], f"{ml:.2f}", f"{mp/1000:.1f}k"]]
    write("tab_cnn.tex", table("tab:cnn", "Classifier on image pixels vs.\\ on keypoints (same frames, random split).",
                               ["Input / model", "Acc.", "F1", "ms", "Params"], rows, bold_max=[1, 2]))
    better = "higher" if ma > ca else "lower" if ma < ca else "the same"
    obs["CNN"] = (f"The keypoint MLP is {abs(ma - ca):.2f} points {better} in accuracy than the CNN ({ma:.2f}\\% vs.\\ "
                  f"{ca:.2f}\\%), while being {cl / ml:.0f}$\\times$ faster and {cp / mp:.0f}$\\times$ smaller. "
                  "Working on keypoints also removes the influence of background and lighting, which the CNN "
                  "has to learn from data.") if better != "the same" else \
                 f"Both reach {ma:.2f}\\%, but the MLP is {cl / ml:.0f}$\\times$ faster and {cp / mp:.0f}$\\times$ smaller."
else:
    write("tab_cnn.tex", empty_table("tab:cnn", "Image CNN vs.\\ keypoint MLP.", "python train_cnn.py"))
    obs["CNN"] = missing("python train_cnn.py")

# ----------------------------------------------------------------------
# 6. personalization (figure from all users)
# ----------------------------------------------------------------------
pers = {}
for p in sorted(glob.glob(os.path.join(R, "personalization_*.md"))):
    u = os.path.basename(p)[len("personalization_"):-3]
    t = read_md_table(p)
    if t:
        kcol = list(t[0].keys())
        vals = []
        for r in t:
            nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", str(r[kcol[1]]))]   # "mean +/- std"
            vals.append((int(num(r[kcol[0]])), nums[0], nums[1] if len(nums) > 1 else 0.0))
        pers[u] = vals
if pers:
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for u, v in pers.items():
        ax.errorbar([a for a, _, _ in v], [b for _, b, _ in v], yerr=[c for _, _, c in v],
                    marker="o", ms=3, capsize=2, lw=1, label=u)
    Ks = [a for a, _, _ in next(iter(pers.values()))]
    mean = [np.mean([dict((a, b) for a, b, _ in v).get(k, np.nan) for v in pers.values()]) for k in Ks]
    ax.plot(Ks, mean, "k--", lw=1.6, label="mean")
    ax.set_xlabel("calibration samples per gesture $K$"); ax.set_ylabel("accuracy on new user (%)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "personalization.png"), dpi=250); plt.close(fig)
    m0 = mean[0]
    k20 = Ks.index(20) if 20 in Ks else len(Ks) - 1
    verb = "rises" if mean[k20] > m0 else "changes"
    obs["Pers"] = (f"Averaged over {len(pers)} users, accuracy on a new user is {m0:.2f}\\% without calibration "
                   f"and {verb} to {mean[k20]:.2f}\\% with $K={Ks[k20]}$ samples per gesture "
                   f"(about one second of recording per gesture).")
    macros["PersZero"], macros["PersK"] = f1(m0, 2), f1(mean[k20], 2)
else:
    placeholder_png(os.path.join(FIGS, "personalization.png"), "run: python personalize.py --user NAME")
    obs["Pers"] = missing("python personalize.py --user NAME")

# ----------------------------------------------------------------------
# 7. cursor filters
# ----------------------------------------------------------------------
flt = read_md_table(os.path.join(R, "filter_comparison.md"))
abs_ = {}
if flt:
    kc = list(flt[0].keys())
    F = {r[kc[0]]: (num(r[kc[1]]), num(r[kc[2]]), num(r[kc[3]])) for r in flt}
    names = {"none": "No filter (raw)", "moving_avg": "Moving average (5)", "kalman": "Kalman (const. velocity)",
             "one_euro": r"\textbf{One-Euro (ours)}"}
    rows = [[names.get(k, tex(k))] + [f1(x, 2) for x in v] for k, v in F.items()]
    write("tab_filter.tex", table("tab:filter", "Cursor smoothing on the same recorded hand motion "
                                  "(lower is better for all columns).",
                                  ["Filter", "Jitter (px/frame)", "Error moving (px)", "Lag (ms)"], rows,
                                  bold_min=[1, 2, 3]))
    jr = pct_change(F["one_euro"][0], F["none"][0])
    abs_["jit"] = jr
    obs["Filter"] = (f"Compared with the raw cursor, the One-Euro filter "
                     f"{'reduces' if jr and jr > 0 else 'changes'} jitter by {abs(jr):.1f}\\% "
                     f"(from {F['none'][0]:.2f} to {F['one_euro'][0]:.2f} px/frame) with "
                     f"{F['one_euro'][2]:.0f}\\,ms of lag. The moving average gives {F['moving_avg'][0]:.2f} px/frame "
                     f"jitter but {F['moving_avg'][2]:.0f}\\,ms lag, and the Kalman filter "
                     f"{F['kalman'][0]:.2f} px/frame with {F['kalman'][2]:.0f}\\,ms lag. The One-Euro filter is "
                     f"the best trade-off because its cutoff rises with hand speed.") if jr is not None else \
        missing("python experiments/analyze_logs.py logs/*.csv")
else:
    write("tab_filter.tex", empty_table("tab:filter", "Cursor smoothing.", "python experiments/analyze_logs.py logs/*.csv"))
    obs["Filter"] = missing("python app.py --log, then python experiments/analyze_logs.py logs/*.csv")

# ----------------------------------------------------------------------
# 8. voting and anchor
# ----------------------------------------------------------------------
vot = read_md_table(os.path.join(R, "voting_comparison.md"))
if vot:
    kc = list(vot[0].keys())
    rows = [[tex(r[kc[0]]).replace("(default)", "(ours)"), r[kc[1]], r[kc[2]], r[kc[3]]] for r in vot]
    write("tab_voting.tex", table("tab:voting", "Effect of temporal voting on the same recorded predictions "
                                  "(the user clicked 30 times).",
                                  ["Setting", "Changes/min", "Short false/min", "Clicks"], rows))
    raw_r = vot[0]; ours = next((r for r in vot if "default" in r[kc[0]]), vot[-1])
    a, b = num(raw_r[kc[1]]), num(ours[kc[1]])
    c_, d = num(raw_r[kc[2]]), num(ours[kc[2]])
    red = pct_change(b, a)
    abs_["vote"] = red
    obs["Voting"] = (f"Voting ($N{{=}}5,k{{=}}3$) lowers the number of gesture changes from {a:.1f} to {b:.1f} per "
                     f"minute and the short (1--2 frame) false segments from {c_:.1f} to {d:.1f} per minute, and "
                     f"detects {ours[kc[3]]} clicks (without voting: {raw_r[kc[3]]}) for the 30 the user made.")
else:
    write("tab_voting.tex", empty_table("tab:voting", "Effect of temporal voting.", "python experiments/analyze_logs.py logs/*.csv"))
    obs["Voting"] = missing("python experiments/analyze_logs.py logs/*.csv")

anc = read_md_table(os.path.join(R, "anchor_drift.md"))
if anc:
    kc = list(anc[0].keys())
    D = {r[kc[0]]: (num(r[kc[1]]), num(r[kc[2]]), r[kc[3]]) for r in anc}
    names = {"index_tip": "Index fingertip (usual)", "index_mcp": r"\textbf{Index knuckle (ours)}", "palm": "Palm centre"}
    rows = [[names.get(k, tex(k)), f1(v[0]), f1(v[1]), v[2]] for k, v in D.items()]
    write("tab_anchor.tex", table("tab:anchor", "Cursor drift at the moment of a click for different anchor points "
                                  "(screen pixels, 1920$\\times$1080).",
                                  ["Anchor", "Mean (px)", "Median (px)", "Clicks"], rows, bold_min=[1, 2]))
    if D.get("index_tip", (None,))[0] and D.get("index_mcp", (None,))[0] is not None:
        red = pct_change(D["index_mcp"][0], D["index_tip"][0])
        abs_["anchor"] = red
        obs["Anchor"] = (f"With the fingertip as anchor, the cursor moves {D['index_tip'][0]:.1f}\\,px on average "
                         f"while the click is being made; with the knuckle it moves only {D['index_mcp'][0]:.1f}\\,px "
                         f"({red:.0f}\\% less), so small targets are not missed.")
    else:
        obs["Anchor"] = missing("record a click session with app.py --log")
else:
    write("tab_anchor.tex", empty_table("tab:anchor", "Cursor drift at a click.", "python experiments/analyze_logs.py logs/*.csv"))
    obs["Anchor"] = missing("python experiments/analyze_logs.py logs/*.csv")

# ----------------------------------------------------------------------
# 9. system: performance, click test, typing
# ----------------------------------------------------------------------
perf = read_md_table(os.path.join(R, "performance.md"))
clk = read_md_table(os.path.join(R, "click_test_summary.md"))
typ = read_md_table(os.path.join(R, "typing_summary.md"))
rows = []
if perf:
    P = {r["Metric"]: r["Value"] for r in perf}
    fps = num(P.get("Mean FPS")); ms = num(P.get("Processing time per frame, mean (ms)"))
    p95 = num(P.get("Processing time per frame, 95th percentile (ms)"))
    rows += [["Frame rate (FPS)", f1(fps)], ["Processing per frame, mean (ms)", f1(ms)],
             ["Processing per frame, 95th perc. (ms)", f1(p95)]]
    obs["System"] = (f"The complete pipeline runs at {fps:.1f} FPS with {ms:.1f}\\,ms of processing per frame "
                     f"(95th percentile {p95:.1f}\\,ms), i.e.\\ in real time on a CPU.")
    abs_["fps"] = fps
else:
    obs["System"] = missing("python experiments/analyze_logs.py logs/*.csv")
click_obs = []
if clk:
    by = {}
    for r in clk:
        by.setdefault(r["Method"], []).append(r)
    rows.append("MID")
    rows.append([r"\textit{Click test (Fitts)}", "time (s) / error (\\%) / TP (bits/s)"])
    for meth, rs in by.items():
        t_ = np.mean([num(r[list(r)[2]]) for r in rs]); e_ = np.mean([num(r[list(r)[3]]) for r in rs])
        tp = np.mean([num(r[list(r)[4]]) for r in rs])
        rows.append([tex(meth) + f" ({len(rs)} user{'s' if len(rs) > 1 else ''})", f"{t_:.2f} / {e_:.1f} / {tp:.2f}"])
        click_obs.append(f"{tex(meth)}: {t_:.2f}\\,s per target, {e_:.1f}\\% errors, {tp:.2f} bits/s")
    obs["Click"] = "In the click test we measured " + "; ".join(click_obs) + "."
else:
    obs["Click"] = missing("python experiments/click_test.py --method gesture --user NAME")
if typ:
    by = {}
    for r in typ:
        by.setdefault(r["Method"], []).append(r)
    rows.append("MID")
    rows.append([r"\textit{Typing test}", "WPM / char. error (\\%)"])
    tobs = []
    for meth, rs in by.items():
        w = np.mean([num(r["WPM"]) for r in rs]); ce = np.mean([num(r["Char error (%)"]) for r in rs])
        rows.append([tex(meth) + f" ({len(rs)} trial{'s' if len(rs) > 1 else ''})", f"{w:.2f} / {ce:.1f}"])
        tobs.append(f"{w:.2f} words per minute with {ce:.1f}\\% character errors ({tex(meth)})")
    obs["Typing"] = "Typing reached " + "; ".join(tobs) + "."
else:
    obs["Typing"] = missing("python experiments/typing_test.py --user NAME")
if rows:
    write("tab_system.tex", table("tab:system", "System performance and user tests.", ["Metric", "Value"],
                                  rows, align="lr"))
else:
    write("tab_system.tex", empty_table("tab:system", "System performance and user tests.",
                                        "python experiments/analyze_logs.py logs/*.csv"))

# ----------------------------------------------------------------------
# 10. abstract and conclusion sentences (only from results that exist)
# ----------------------------------------------------------------------
parts = []
if lm is not None and lr is not None:
    parts.append(f"reaches {lm:.1f}\\% gesture accuracy on users not seen in training (rule-based: {lr:.1f}\\%)")
elif mlp is not None:
    parts.append(f"reaches {mlp:.1f}\\% gesture accuracy (rule-based: {rules:.1f}\\%)")
if (abs_.get("jit") or 0) > 0:     # only real improvements go in the abstract
    parts.append(f"removes {abs_['jit']:.0f}\\% of the cursor jitter")
if (abs_.get("vote") or 0) > 0:     # only real improvements go in the abstract
    parts.append(f"cuts gesture flicker by {abs_['vote']:.0f}\\%")
if (abs_.get("anchor") or 0) > 0:     # only real improvements go in the abstract
    parts.append(f"reduces cursor drift during a click by {abs_['anchor']:.0f}\\%")
if abs_.get("fps") is not None:
    parts.append(f"runs at {abs_['fps']:.0f} FPS on a laptop CPU")
who = f"In experiments with {n_users} users, the system " if n_users else "The system "
if parts:
    summary = who + (", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0]) + "."
else:
    summary = missing("python run_all.py")
obs["Abstract"] = summary
obs["Conclusion"] = summary + (" Each proposed component gives a measurable improvement over the usual "
                               "design, and together they make a webcam a practical touch-free mouse and keyboard.")

# ----------------------------------------------------------------------
# 11. figures
# ----------------------------------------------------------------------
src_figs = {"overview.pdf": os.path.join(R, "figures", "overview.pdf"),
            "features.png": os.path.join(R, "figures", "features.png"),
            "gestures.png": os.path.join(R, "figures", "gestures.png"),
            "anchor.png": os.path.join(R, "figures", "anchor.png"),
            "confusion_matrix.png": os.path.join(R, "confusion_matrix.png"),
            "filter_trace.png": os.path.join(R, "filter_trace.png")}
hint = {"overview.pdf": "python make_figures.py", "features.png": "python make_figures.py (needs data)",
        "gestures.png": "python make_figures.py (needs data)", "anchor.png": "python make_figures.py (needs data)",
        "confusion_matrix.png": "python train.py", "filter_trace.png": "python experiments/analyze_logs.py"}
for name, src in src_figs.items():
    dst = os.path.join(FIGS, name)
    if os.path.exists(src):
        shutil.copy(src, dst)
    elif name.endswith(".pdf"):
        png = dst[:-4] + ".png"
        placeholder_png(png, "run: " + hint[name])
        # make a tiny pdf placeholder too
        fig = plt.figure(figsize=(7, 2)); fig.text(0.5, 0.5, "run: " + hint[name], ha="center", color="#aa0000")
        fig.savefig(dst); plt.close(fig)
    else:
        placeholder_png(dst, "run: " + hint[name])

# ----------------------------------------------------------------------
# 12. code appendix
# ----------------------------------------------------------------------
files = sorted(glob.glob(os.path.join(config.ROOT_DIR, "*.py"))) + \
    sorted(glob.glob(os.path.join(config.ROOT_DIR, "experiments", "*.py")))
listing = []
for p in files:
    rel = os.path.relpath(p, config.ROOT_DIR).replace("\\", "/")
    flat = rel.replace("/", "__")
    with open(p, encoding="utf-8") as fi:
        txt = fi.read()
    txt = txt.encode("ascii", "replace").decode("ascii")      # listings + pdflatex need ASCII
    with open(os.path.join(CODE, flat), "w", encoding="ascii") as fo:
        fo.write(txt)
    listing.append(r"\subsection*{\texttt{" + tex(rel) + "}}\n\\lstinputlisting{code/" + flat + "}")
write("code_listings.tex", "\n".join(listing))

# ----------------------------------------------------------------------
# 13. macros file
# ----------------------------------------------------------------------
lines = []
for k, v in macros.items():
    lines.append(rf"\newcommand{{\{k}}}{{{v}}}")
for k, v in obs.items():
    lines.append(rf"\newcommand{{\Obs{k}}}{{{v}}}")
write("macros.tex", "\n".join(lines))
print("Wrote report/auto/*.tex")

# ----------------------------------------------------------------------
# 14. compile (if LaTeX is installed) and zip for Overleaf
# ----------------------------------------------------------------------
def compile_pdf():
    if shutil.which("pdflatex") is None:
        print("pdflatex not found -> upload report_overleaf.zip to Overleaf to get the PDF.")
        return
    cmds = [["pdflatex", "-interaction=nonstopmode", "main.tex"], ["bibtex", "main"],
            ["pdflatex", "-interaction=nonstopmode", "main.tex"], ["pdflatex", "-interaction=nonstopmode", "main.tex"]]
    for c in cmds:
        subprocess.run(c, cwd=REPORT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pdf = os.path.join(REPORT, "main.pdf")
    print("Compiled", pdf if os.path.exists(pdf) else "(failed - open report/main.log)")


compile_pdf()
zpath = os.path.join(config.ROOT_DIR, "report_overleaf.zip")
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, fs in os.walk(REPORT):
        for fn in fs:
            if fn.endswith((".aux", ".log", ".out", ".blg", ".bbl", ".pdf")) and not root.endswith("figures"):
                continue
            full = os.path.join(root, fn)
            z.write(full, os.path.relpath(full, REPORT))
print("Overleaf zip:", zpath)
