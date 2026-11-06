"""
train.py
--------
Train the gesture MLP and run all classification experiments.

    python train.py                     # random 80/20 split, ablation + baselines
    python train.py --test-user guna    # train on the other users, test on 'guna'
    python train.py --loso              # leave-one-user-out over every user

Outputs (in results/):
    ablation.md / ablation.csv     table: feature mode, augmentation, baselines
    loso.md                        table: accuracy on unseen users (with --loso)
    confusion_matrix.png           confusion matrix of the final MLP
    training_curve.png             loss / accuracy per epoch
    classification_report.txt      precision / recall / F1 per gesture
    metrics.json                   all numbers in one file
and the deployed model in models/gesture_mlp.pt
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import config
from features import extract_features, augment_landmarks, FEATURE_MODES
from model import GestureMLP, save_model, count_params
from rules import classify_rules


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_dataset(path=config.LANDMARK_CSV):
    """Returns users (N,), labels (N,), handedness (N,), landmarks (N,21,3)."""
    users, labels, hands, lms = [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["label"] not in config.GESTURES:
                continue
            users.append(row["user"])
            labels.append(config.GESTURES.index(row["label"]))
            hands.append(row["handedness"])
            lms.append([float(row[f"{a}{i}"]) for i in range(21) for a in "xyz"])
    if not labels:
        raise SystemExit(f"No data in {path}. Run collect_data.py first.")
    return (np.array(users), np.array(labels), np.array(hands),
            np.array(lms, dtype=np.float32).reshape(-1, 21, 3))


def featurize(lms, hands, mode):
    return np.stack([extract_features(l, h, mode) for l, h in zip(lms, hands)]).astype(np.float32)


def augment(lms, hands, labels, copies, seed=config.SEED):
    """Add `copies` randomly-augmented versions of every training sample."""
    if copies <= 0:
        return lms, hands, labels
    rng = np.random.default_rng(seed)
    aug = [augment_landmarks(l, rng) for _ in range(copies) for l in lms]
    return (np.concatenate([lms, np.stack(aug)]),
            np.concatenate([hands] + [hands] * copies),
            np.concatenate([labels] + [labels] * copies))


# --------------------------------------------------------------------------
# MLP training
# --------------------------------------------------------------------------
def train_mlp(Xtr, ytr, Xval, yval, epochs=config.EPOCHS, verbose=True, seed=config.SEED):
    """Train on standardized features. Keeps the weights of the best validation epoch."""
    torch.manual_seed(seed)
    model = GestureMLP(Xtr.shape[1], config.NUM_CLASSES)
    opt = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr).long()
    Xval_t, yval_t = torch.from_numpy(Xval), torch.from_numpy(yval).long()

    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc, best_state = -1, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        tot = 0.0
        for i in range(0, len(perm), config.BATCH_SIZE):
            idx = perm[i:i + config.BATCH_SIZE]
            if len(idx) < 2:          # BatchNorm needs >1 sample
                continue
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()

        model.eval()
        with torch.no_grad():
            out = model(Xval_t)
            vloss = loss_fn(out, yval_t).item()
            vacc = (out.argmax(1) == yval_t).float().mean().item()
        hist["train_loss"].append(tot / len(Xtr_t))
        hist["val_loss"].append(vloss)
        hist["val_acc"].append(vacc)
        if vacc > best_acc:
            best_acc = vacc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  train_loss {hist['train_loss'][-1]:.4f}  "
                  f"val_loss {vloss:.4f}  val_acc {vacc:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    return model, hist


@torch.no_grad()
def mlp_predict(model, X):
    return model(torch.from_numpy(X)).argmax(1).numpy()


def time_per_sample(fn, n=300):
    """Average time (ms) for one call of fn() - single-sample latency as in the live app."""
    fn()  # warm-up
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000


def run_mlp_experiment(lm_tr, h_tr, y_tr, lm_te, h_te, y_te, mode, aug_copies, verbose=False):
    """Full pipeline for one MLP setting. Returns (model, mean, std, metrics, history)."""
    # validation split (10 %) taken from the training users only
    idx_tr, idx_val = train_test_split(np.arange(len(y_tr)), test_size=0.1,
                                       random_state=config.SEED, stratify=y_tr)
    a_lm, a_h, a_y = augment(lm_tr[idx_tr], h_tr[idx_tr], y_tr[idx_tr], aug_copies)
    Xtr = featurize(a_lm, a_h, mode)
    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-6
    norm = lambda X: ((X - mean) / std).astype(np.float32)
    Xval = norm(featurize(lm_tr[idx_val], h_tr[idx_val], mode))
    Xte = norm(featurize(lm_te, h_te, mode))

    model, hist = train_mlp(norm(Xtr), a_y, Xval, y_tr[idx_val], verbose=verbose)
    pred = mlp_predict(model, Xte)

    # latency: feature extraction + forward pass for ONE sample
    torch.set_num_threads(1)
    l0, h0 = lm_te[0], h_te[0]
    def one():
        f = norm(extract_features(l0, h0, mode)[None])
        with torch.no_grad():
            model(torch.from_numpy(f))
    ms = time_per_sample(one)
    torch.set_num_threads(os.cpu_count() or 1)

    m = {"acc": accuracy_score(y_te, pred), "macro_f1": f1_score(y_te, pred, average="macro"),
         "latency_ms": ms, "params": count_params(model)}
    return model, mean, std, m, hist, pred


def run_sklearn(clf, lm_tr, h_tr, y_tr, lm_te, h_te, y_te):
    Xtr, Xte = featurize(lm_tr, h_tr, "full"), featurize(lm_te, h_te, "full")
    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf.fit((Xtr - mean) / std, y_tr)
    pred = clf.predict((Xte - mean) / std)
    x0 = ((extract_features(lm_te[0], h_te[0], "full") - mean) / std)[None]
    ms = time_per_sample(lambda: clf.predict(x0), n=100)
    return {"acc": accuracy_score(y_te, pred), "macro_f1": f1_score(y_te, pred, average="macro"),
            "latency_ms": ms}


def run_rules(lm_te, h_te, y_te):
    pred = np.array([classify_rules(l, h)[0] for l, h in zip(lm_te, h_te)])
    ms = time_per_sample(lambda: classify_rules(lm_te[0], h_te[0]))
    return {"acc": accuracy_score(y_te, pred), "macro_f1": f1_score(y_te, pred, average="macro"),
            "latency_ms": ms}


# --------------------------------------------------------------------------
# Plots and tables
# --------------------------------------------------------------------------
def plot_confusion(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=range(config.NUM_CLASSES))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(config.NUM_CLASSES)); ax.set_yticks(range(config.NUM_CLASSES))
    ax.set_xticklabels(config.GESTURES, rotation=45, ha="right"); ax.set_yticklabels(config.GESTURES)
    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            ax.text(j, i, f"{cmn[i, j]*100:.0f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_history(hist, path):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
    ax[1].plot(hist["val_acc"]); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val accuracy")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def write_table(rows, path_md, path_csv, cols):
    with open(path_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])
    with open(path_md, "w") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            cells = []
            for c in cols:
                v = r.get(c, "")
                cells.append(f"{v*100:.2f}" if c in ("acc", "macro_f1") and v != "" else
                             f"{v:.3f}" if isinstance(v, float) else str(v))
            f.write("| " + " | ".join(cells) + " |\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def split(users, labels, test_user=None):
    if test_user:
        te = users == test_user
        if te.sum() == 0:
            raise SystemExit(f"No samples for user {test_user}")
        return np.where(~te)[0], np.where(te)[0]
    return train_test_split(np.arange(len(labels)), test_size=0.2,
                            random_state=config.SEED, stratify=labels)


def ablation(users, labels, hands, lms, test_user=None):
    tr, te = split(users, labels, test_user)
    args = (lms[tr], hands[tr], labels[tr], lms[te], hands[te], labels[te])
    print(f"Train samples: {len(tr)}   Test samples: {len(te)}   "
          f"(test = {'user ' + test_user if test_user else 'random 20%'})")
    rows, results = [], {}

    print("\n[1] Rule-based baseline (finger-up heuristics)")
    r = run_rules(lms[te], hands[te], labels[te]); r["method"] = "Rule-based (fingers up)"
    rows.append(r)

    for name, clf in [("SVM (RBF)", SVC(C=10, gamma="scale")),
                      ("Random Forest", RandomForestClassifier(200, random_state=config.SEED)),
                      ("kNN (k=5)", KNeighborsClassifier(5))]:
        print(f"[2] {name} on full features")
        r = run_sklearn(clf, *args); r["method"] = f"{name}, full features"; rows.append(r)

    final = None
    for mode in FEATURE_MODES:
        for aug in ([0, config.AUGMENT_COPIES] if mode == "full" else [config.AUGMENT_COPIES]):
            print(f"[3] MLP  features={mode}  augmentation copies={aug}")
            model, mean, std, m, hist, pred = run_mlp_experiment(*args, mode, aug,
                                                                 verbose=(mode == "full" and aug > 0))
            m["method"] = f"MLP, {mode} features" + ("" if aug else ", no augmentation")
            rows.append(m)
            if mode == "full" and aug > 0:
                final = (model, mean, std, hist, pred, m)

    for r in rows:
        print(f"  {r['method']:<40} acc={r['acc']*100:6.2f}%  F1={r['macro_f1']*100:6.2f}%  "
              f"latency={r['latency_ms']:.3f} ms")
    cols = ["method", "acc", "macro_f1", "latency_ms", "params"]
    tag = f"_{test_user}" if test_user else ""
    write_table(rows, os.path.join(config.RESULTS_DIR, f"ablation{tag}.md"),
                os.path.join(config.RESULTS_DIR, f"ablation{tag}.csv"), cols)

    model, mean, std, hist, pred, m = final
    save_model(model, config.GESTURE_MODEL_PATH, "full", mean, std,
               extra={"test_acc": m["acc"], "test_user": test_user})
    plot_confusion(labels[te], pred, os.path.join(config.RESULTS_DIR, f"confusion_matrix{tag}.png"),
                   f"MLP (full features) - acc {m['acc']*100:.1f}%")
    plot_history(hist, os.path.join(config.RESULTS_DIR, f"training_curve{tag}.png"))
    with open(os.path.join(config.RESULTS_DIR, f"classification_report{tag}.txt"), "w") as f:
        f.write(classification_report(labels[te], pred, labels=range(config.NUM_CLASSES),
                                      target_names=config.GESTURES, digits=4, zero_division=0))
    results["rows"] = rows
    print(f"\nSaved model to {config.GESTURE_MODEL_PATH}")
    return results


def loso(users, labels, hands, lms):
    """Leave-one-user-out: how well does it work for a person not in the training data?"""
    rows = []
    for u in sorted(set(users)):
        tr, te = split(users, labels, u)
        args = (lms[tr], hands[tr], labels[tr], lms[te], hands[te], labels[te])
        print(f"\n=== test user: {u}  (train {len(tr)}, test {len(te)}) ===")
        r_rule = run_rules(lms[te], hands[te], labels[te])
        r_svm = run_sklearn(SVC(C=10, gamma="scale"), *args)
        _, _, _, r_mlp, _, _ = run_mlp_experiment(*args, "full", config.AUGMENT_COPIES)
        _, _, _, r_raw, _, _ = run_mlp_experiment(*args, "raw", 0)
        rows.append({"test_user": str(u), "rules": r_rule["acc"], "svm": r_svm["acc"],
                     "mlp_raw": r_raw["acc"], "mlp_full": r_mlp["acc"]})
        print(rows[-1])
    mean = {k: float(np.mean([r[k] for r in rows])) for k in ("rules", "svm", "mlp_raw", "mlp_full")}
    rows.append({"test_user": "MEAN", **mean})
    with open(os.path.join(config.RESULTS_DIR, "loso.md"), "w") as f:
        f.write("| Test user | Rule-based | SVM (full) | MLP (raw, no aug) | MLP (full + aug, ours) |\n"
                "|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['test_user']} | {r['rules']*100:.2f} | {r['svm']*100:.2f} | "
                    f"{r['mlp_raw']*100:.2f} | {r['mlp_full']*100:.2f} |\n")
    print(open(os.path.join(config.RESULTS_DIR, "loso.md")).read())
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-user", default=None)
    ap.add_argument("--loso", action="store_true")
    args = ap.parse_args()

    users, labels, hands, lms = load_dataset()
    print(f"Loaded {len(labels)} samples from users {sorted(set(users))}")
    print("Per class:", {g: int((labels == i).sum()) for i, g in enumerate(config.GESTURES)})

    out = {"n_samples": int(len(labels)), "users": sorted(set(users.tolist()))}
    if args.loso:
        out["loso"] = loso(users, labels, hands, lms)
    else:
        out["ablation"] = ablation(users, labels, hands, lms, args.test_user)["rows"]
    name = "metrics_loso.json" if args.loso else f"metrics{'_' + args.test_user if args.test_user else ''}.json"
    with open(os.path.join(config.RESULTS_DIR, name), "w") as f:
        json.dump(out, f, indent=2, default=float)


if __name__ == "__main__":
    main()
