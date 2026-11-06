"""
run_all.py
----------
Runs EVERY automatic experiment, then builds the figures, the report and the
slides. Run it after all members have recorded data (collect_data.py) and,
optionally, after recording app logs (app.py --log).

    python run_all.py            # everything
    python run_all.py --quick    # skip the slow parts (CNN, leave-one-user-out)

Order matters:
  1. leave-one-user-out          (train.py --loso)
  2. image-CNN baseline           (train_cnn.py)
  3. personalization per user     (personalize.py --user X)
  4. log analysis                 (experiments/analyze_logs.py logs/*.csv)
  5. final model + ablation       (train.py)  -> models/gesture_mlp.pt used by the app
  6. figures, report, slides      (make_figures.py, make_report.py, make_slides.py)
"""
import argparse
import csv
import glob
import os
import subprocess
import sys
import time

import config

PY = sys.executable


def run(args, required=True):
    print("\n" + "=" * 70 + "\n>>> " + " ".join(args) + "\n" + "=" * 70, flush=True)
    t = time.time()
    r = subprocess.run([PY] + args, cwd=config.ROOT_DIR)
    print(f"<<< finished in {time.time() - t:.0f} s (exit code {r.returncode})")
    if r.returncode != 0 and required:
        raise SystemExit(f"Step failed: {' '.join(args)}")
    return r.returncode == 0


def users_in_data():
    if not os.path.exists(config.LANDMARK_CSV):
        raise SystemExit("No data yet. Every member must run: python collect_data.py --user <name>")
    with open(config.LANDMARK_CSV, newline="") as f:
        return sorted({row["user"] for row in csv.DictReader(f)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    users = users_in_data()
    print("Users in dataset:", users)
    ok = {}
    if len(users) >= 2 and not a.quick:
        ok["loso"] = run(["train.py", "--loso"], required=False)
    if not a.quick:
        ok["cnn"] = run(["train_cnn.py"], required=False)
    if len(users) >= 2:
        for u in users:
            ok[f"personalize_{u}"] = run(["personalize.py", "--user", u], required=False)
    logs = sorted(glob.glob(os.path.join(config.LOG_DIR, "*.csv")))
    if logs:
        ok["logs"] = run([os.path.join("experiments", "analyze_logs.py")] + logs, required=False)
    else:
        print("\n(no logs in logs/ - record them with:  python app.py --log --dry-run)")
    ok["train"] = run(["train.py"])            # last: this is the model the app uses
    ok["figures"] = run(["make_figures.py"], required=False)
    ok["report"] = run(["make_report.py"], required=False)
    ok["slides"] = run(["make_slides.py"], required=False)

    print("\nSUMMARY")
    for k, v in ok.items():
        print(f"  {k:<25} {'OK' if v else 'FAILED (see messages above)'}")
    print("\nReport  : report/  (main.pdf if LaTeX is installed, else upload report_overleaf.zip to Overleaf)")
    print("Slides  : presentation/Gesture_Control_EE604.pptx")


if __name__ == "__main__":
    main()
