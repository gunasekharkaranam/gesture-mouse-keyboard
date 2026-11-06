"""
personalize.py
--------------
Novelty 4: quick per-user calibration (few-shot personalization).

Every person makes gestures a little differently. Instead of re-training the
whole network, we keep the learned feature layers fixed and re-train ONLY the
last layer using K samples per gesture from the new user. This takes about
one second on a CPU.

Experiment mode (for the report):
    python personalize.py --user guna
  - base model is trained WITHOUT guna's data
  - guna's data is split in half: first half = calibration pool, second half = test
  - we report test accuracy for K = 0 (no calibration), 5, 10, 20, 50 samples/gesture
  - outputs results/personalization_<user>.md and .png

Deploy mode (for the demo):
    python personalize.py --user guna --deploy
  - fine-tunes the current models/gesture_mlp.pt on ALL of guna's samples
  - saves models/gesture_mlp_guna.pt  (run the app with --model models/gesture_mlp_guna.pt)
"""
import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from model import load_checkpoint, save_model
from train import load_dataset, featurize, run_mlp_experiment, augment

K_LIST = [0, 5, 10, 20, 50]


def finetune_head(model, X, y, epochs=100, lr=5e-3):
    """Re-train only the last linear layer; the body (feature extractor) is frozen."""
    model = copy.deepcopy(model)
    for p in model.body.parameters():
        p.requires_grad = False
    model.eval()                         # keep BatchNorm/Dropout in inference mode
    opt = torch.optim.Adam(model.head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y).long()
    with torch.no_grad():
        H = model.body(Xt)               # features computed once (fast)
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model.head(H), yt).backward()
        opt.step()
    return model


def pick_k_per_class(y, k, rng):
    idx = []
    for c in np.unique(y):
        cand = np.where(y == c)[0]
        idx += rng.choice(cand, size=min(k, len(cand)), replace=False).tolist()
    return np.array(idx)


def experiment(user, repeats=5):
    users, labels, hands, lms = load_dataset()
    tr, te_all = np.where(users != user)[0], np.where(users == user)[0]
    if len(te_all) == 0 or len(tr) == 0:
        raise SystemExit("Need data from this user AND at least one other user.")

    print(f"Training base model without '{user}' ({len(tr)} samples) ...")
    # base model: trained on other users (te arguments only used for its own metrics)
    model, mean, std, m, _, _ = run_mlp_experiment(lms[tr], hands[tr], labels[tr],
                                                   lms[te_all], hands[te_all], labels[te_all],
                                                   "full", config.AUGMENT_COPIES)
    norm = lambda X: ((X - mean) / std).astype(np.float32)

    # split this user's data in time order: first half = calibration pool, second half = test
    pool, test = [], []
    for c in range(config.NUM_CLASSES):
        ids = te_all[labels[te_all] == c]
        pool += ids[: len(ids) // 2].tolist()
        test += ids[len(ids) // 2:].tolist()
    pool, test = np.array(pool), np.array(test)
    Xte = torch.from_numpy(norm(featurize(lms[test], hands[test], "full")))
    yte = labels[test]

    rows = []
    for k in K_LIST:
        accs = []
        for r in range(repeats if k > 0 else 1):
            if k == 0:
                mdl = model
            else:
                rng = np.random.default_rng(r)
                sel = pool[pick_k_per_class(labels[pool], k, rng)]
                a_lm, a_h, a_y = augment(lms[sel], hands[sel], labels[sel], 4, seed=r)
                mdl = finetune_head(model, norm(featurize(a_lm, a_h, "full")), a_y)
            with torch.no_grad():
                accs.append(float((mdl(Xte).argmax(1).numpy() == yte).mean()))
        rows.append((k, np.mean(accs), np.std(accs)))
        print(f"  K={k:3d} samples/gesture -> accuracy {np.mean(accs)*100:.2f} +/- {np.std(accs)*100:.2f}%")

    md = os.path.join(config.RESULTS_DIR, f"personalization_{user}.md")
    with open(md, "w") as f:
        f.write(f"| Calibration samples per gesture (user {user}) | Accuracy (%) |\n|---|---|\n")
        for k, a, s in rows:
            f.write(f"| {k} | {a*100:.2f} +/- {s*100:.2f} |\n")
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.errorbar([r[0] for r in rows], [r[1] * 100 for r in rows], yerr=[r[2] * 100 for r in rows],
                marker="o", capsize=3)
    ax.set_xlabel("calibration samples per gesture (K)"); ax.set_ylabel("accuracy on new user (%)")
    ax.set_title(f"Few-shot personalization ({user})"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(config.RESULTS_DIR, f"personalization_{user}.png"), dpi=200)
    print(f"Saved {md}")


def deploy(user):
    users, labels, hands, lms = load_dataset()
    sel = np.where(users == user)[0]
    if len(sel) == 0:
        raise SystemExit(f"No samples for {user}. Record some with collect_data.py --user {user}")
    model, ckpt = load_checkpoint(config.GESTURE_MODEL_PATH)
    mean, std = np.array(ckpt["mean"]), np.array(ckpt["std"])
    a_lm, a_h, a_y = augment(lms[sel], hands[sel], labels[sel], 4)
    X = ((featurize(a_lm, a_h, "full") - mean) / std).astype(np.float32)
    model = finetune_head(model, X, a_y)
    out = os.path.join(config.MODEL_DIR, f"gesture_mlp_{user}.pt")
    save_model(model, out, "full", mean, std, extra={"personalized_for": user})
    print(f"Saved personalized model to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--deploy", action="store_true")
    a = ap.parse_args()
    deploy(a.user) if a.deploy else experiment(a.user)
