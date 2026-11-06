"""
train_cnn.py
------------
Baseline: a small CNN that classifies the gesture directly from the hand
IMAGE crop (saved by collect_data.py in data/images/<label>/<user>_*.jpg).

We compare it with our keypoint MLP on accuracy, speed and model size,
to justify the choice of keypoint features. (Both still need MediaPipe to
find and crop the hand, so the comparison is only about the classifier.)

    python train_cnn.py                    # random 80/20 split
    python train_cnn.py --test-user guna   # test on an unseen person
Output: results/cnn_baseline.md
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import config

IMG = 64


class SmallCNN(nn.Module):
    def __init__(self, n):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n))

    def forward(self, x):
        return self.head(self.f(x))


def load_images():
    X, y, users = [], [], []
    for ci, g in enumerate(config.GESTURES):
        for p in glob.glob(os.path.join(config.IMAGE_DIR, g, "*.jpg")):
            im = cv2.imread(p)
            if im is None:
                continue
            X.append(cv2.resize(im, (IMG, IMG))[:, :, ::-1])
            y.append(ci)
            users.append(os.path.basename(p).rsplit("_", 1)[0])
    if not X:
        raise SystemExit("No images found. Run collect_data.py (with --save-images 1) first.")
    X = np.stack(X).astype(np.float32) / 255.0
    return X.transpose(0, 3, 1, 2), np.array(y), np.array(users)


def augment_batch(x):
    """Random brightness / contrast and small shifts (training only)."""
    b = x.shape[0]
    x = x * torch.empty(b, 1, 1, 1).uniform_(0.7, 1.3) + torch.empty(b, 1, 1, 1).uniform_(-0.1, 0.1)
    dx, dy = np.random.randint(-4, 5, size=2)
    x = torch.roll(x, shifts=(int(dy), int(dx)), dims=(2, 3))
    return x.clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-user", default=None)
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()
    torch.manual_seed(config.SEED); np.random.seed(config.SEED)

    X, y, users = load_images()
    print(f"{len(y)} images, users {sorted(set(users))}")
    if args.test_user:
        tr, te = np.where(users != args.test_user)[0], np.where(users == args.test_user)[0]
    else:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=config.SEED, stratify=y)

    model = SmallCNN(config.NUM_CLASSES)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    Xtr, ytr = torch.from_numpy(X[tr]), torch.from_numpy(y[tr]).long()
    Xte = torch.from_numpy(X[te])
    t_start = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot = 0
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = loss_fn(model(augment_batch(Xtr[idx])), ytr[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        print(f"epoch {ep:2d} loss {tot/len(Xtr):.4f}")
    train_time = time.time() - t_start

    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(Xte[i:i + 256]).argmax(1) for i in range(0, len(Xte), 256)]).numpy()
    acc, f1 = accuracy_score(y[te], pred), f1_score(y[te], pred, average="macro")

    torch.set_num_threads(1)
    x0 = Xte[:1]
    with torch.no_grad():
        model(x0)
        t0 = time.perf_counter()
        for _ in range(100):
            model(x0)
    ms = (time.perf_counter() - t0) / 100 * 1000
    params = sum(p.numel() for p in model.parameters())

    tag = f"_{args.test_user}" if args.test_user else ""
    out = os.path.join(config.RESULTS_DIR, f"cnn_baseline{tag}.md")
    with open(out, "w") as f:
        f.write("| Method | Accuracy (%) | Macro F1 (%) | Latency per sample (ms) | Parameters | Train time (s) |\n"
                "|---|---|---|---|---|---|\n")
        f.write(f"| CNN on {IMG}x{IMG} hand image | {acc*100:.2f} | {f1*100:.2f} | {ms:.2f} | {params} | {train_time:.0f} |\n")
    print(open(out).read())
    torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, "gesture_cnn.pt"))


if __name__ == "__main__":
    main()
