"""
voting.py
---------
Confidence + temporal voting ("debouncing") of the per-frame gesture predictions.

The classifier sometimes flips to a wrong class for one or two frames
(for example while the hand is moving from "point" to "pinch").
If we acted on every frame, these flips would create unwanted clicks.

Rule: the stable gesture only changes to class c when, among the last
VOTE_WINDOW frames, c was predicted at least VOTE_MIN_COUNT times AND the
mean confidence of those predictions is above CONF_THRESHOLD.
Otherwise the previous stable gesture is kept.
"""
from collections import deque, Counter

import config


class GestureVoter:
    def __init__(self, window=config.VOTE_WINDOW, min_count=config.VOTE_MIN_COUNT,
                 conf_thr=config.CONF_THRESHOLD, enabled=True):
        self.window, self.min_count, self.conf_thr = window, min_count, conf_thr
        self.enabled = enabled
        self.buf = deque(maxlen=window)
        self.stable = config.GESTURES.index("idle_palm")

    def reset(self):
        self.buf.clear()
        self.stable = config.GESTURES.index("idle_palm")

    def update(self, label, conf):
        """Feed one frame's prediction; returns the stable label."""
        if not self.enabled:
            self.stable = label
            return label
        self.buf.append((label, conf))
        counts = Counter(l for l, _ in self.buf)
        best, n = counts.most_common(1)[0]
        mean_conf = sum(c for l, c in self.buf if l == best) / n
        if n >= self.min_count and mean_conf >= self.conf_thr:
            self.stable = best
        return self.stable
