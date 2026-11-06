"""
filters.py
----------
Cursor smoothing filters. Hand keypoints shake a little in every frame
(jitter), so a raw cursor is hard to use for small targets. We compare:

  none        : raw position
  moving_avg  : mean of the last N positions (smooth but laggy)
  kalman      : constant-velocity Kalman filter
  one_euro    : One-Euro filter (Casiez et al., CHI 2012) - low cutoff when the
                hand is slow (removes jitter), high cutoff when fast (removes lag).
                This speed-adaptive behaviour is what we use by default.

All filters work on 2-D points in SCREEN PIXELS and take the time stamp t (seconds).
(The One-Euro beta and the Kalman noise values in config.py assume pixel units.)
"""
import math
from collections import deque

import numpy as np

import config


class NoFilter:
    name = "none"

    def __call__(self, x, y, t):
        return x, y

    def reset(self):
        pass


class MovingAverage:
    name = "moving_avg"

    def __init__(self, window=config.MOVING_AVG_WINDOW):
        self.buf = deque(maxlen=window)

    def __call__(self, x, y, t):
        self.buf.append((x, y))
        a = np.mean(self.buf, axis=0)
        return float(a[0]), float(a[1])

    def reset(self):
        self.buf.clear()


class Kalman2D:
    """State = [x, y, vx, vy], constant-velocity model."""
    name = "kalman"

    def __init__(self, q=config.KALMAN_PROCESS_NOISE, r=config.KALMAN_MEASURE_NOISE):
        self.q, self.r = q, r
        self.reset()

    def reset(self):
        self.s = None
        self.P = np.eye(4)
        self.t = None

    def __call__(self, x, y, t):
        z = np.array([x, y])
        if self.s is None:
            self.s = np.array([x, y, 0.0, 0.0])
            self.t = t
            return x, y
        dt = max(t - self.t, 1e-3)
        self.t = t
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        Q = self.q * np.eye(4)
        R = self.r * np.eye(2)
        # predict
        self.s = F @ self.s
        self.P = F @ self.P @ F.T + Q
        # update
        y_res = z - H @ self.s
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.s = self.s + K @ y_res
        self.P = (np.eye(4) - K @ H) @ self.P
        return float(self.s[0]), float(self.s[1])


class _LowPass:
    def __init__(self):
        self.y = None

    def __call__(self, x, alpha):
        self.y = x if self.y is None else alpha * x + (1 - alpha) * self.y
        return self.y


class OneEuro1D:
    def __init__(self, min_cutoff, beta, d_cutoff):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_f, self.dx_f = _LowPass(), _LowPass()
        self.t_prev, self.x_prev = None, None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t_prev is None:
            self.t_prev, self.x_prev = t, x
            self.dx_f(0.0, 1.0)
            return self.x_f(x, 1.0)
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt                       # speed
        edx = self.dx_f(dx, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)   # faster hand -> higher cutoff
        out = self.x_f(x, self._alpha(cutoff, dt))
        self.t_prev, self.x_prev = t, x
        return out


class OneEuro:
    name = "one_euro"

    def __init__(self, min_cutoff=config.ONE_EURO_MIN_CUTOFF, beta=config.ONE_EURO_BETA,
                 d_cutoff=config.ONE_EURO_D_CUTOFF):
        self.args = (min_cutoff, beta, d_cutoff)
        self.reset()

    def reset(self):
        self.fx, self.fy = OneEuro1D(*self.args), OneEuro1D(*self.args)

    def __call__(self, x, y, t):
        return self.fx(x, t), self.fy(y, t)


FILTERS = {"none": NoFilter, "moving_avg": MovingAverage, "kalman": Kalman2D, "one_euro": OneEuro}


def make_filter(name):
    return FILTERS[name]()
