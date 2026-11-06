"""
model.py
--------
The deep-learning part: a small multi-layer perceptron (MLP) that classifies
the hand-keypoint feature vector into one of the gesture classes.

Why an MLP on keypoints instead of a CNN on images?
  * the input is only 88 numbers, so the network is tiny (~20k parameters)
  * it runs in well under 1 ms on a CPU, leaving time for real-time control
  * it ignores background, skin colour and lighting (MediaPipe already handled that)
The CNN-on-images baseline in train_cnn.py lets us check this claim.
"""
import json

import numpy as np
import torch
import torch.nn as nn

import config
from features import extract_features


class GestureMLP(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=(128, 64), dropout=0.2):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_classes)   # last layer (re-trained in personalization)

    def forward(self, x):
        return self.head(self.body(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def save_model(model, path, feature_mode, mean, std, extra=None):
    """Save weights + everything needed to use the model later."""
    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": model.body[0].in_features,
        "hidden": [m.out_features for m in model.body if isinstance(m, nn.Linear)],
        "num_classes": model.head.out_features,
        "feature_mode": feature_mode,
        "mean": np.asarray(mean, dtype=np.float32).tolist(),
        "std": np.asarray(std, dtype=np.float32).tolist(),
        "classes": config.GESTURES,
        "extra": extra or {},
    }, path)


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = GestureMLP(ckpt["in_dim"], ckpt["num_classes"], hidden=tuple(ckpt["hidden"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


class GesturePredictor:
    """Used by the live app: landmarks in -> (class index, confidence) out."""

    def __init__(self, path=config.GESTURE_MODEL_PATH):
        self.model, ckpt = load_checkpoint(path)
        self.mode = ckpt["feature_mode"]
        self.mean = np.array(ckpt["mean"], dtype=np.float32)
        self.std = np.array(ckpt["std"], dtype=np.float32)
        torch.set_num_threads(1)     # tiny model: 1 thread is fastest

    @torch.no_grad()
    def predict(self, landmarks, handedness="Right"):
        f = extract_features(landmarks, handedness, self.mode)
        x = torch.from_numpy(((f - self.mean) / self.std)[None, :].astype(np.float32))
        prob = torch.softmax(self.model(x), dim=1)[0].numpy()
        k = int(prob.argmax())
        return k, float(prob[k])
