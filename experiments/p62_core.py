"""核心：16×16 温度计编码 + 大前额叶（局部随机三层 + top-m 最后一层）。"""
from __future__ import annotations

import numpy as np

G = 16
LV = 4
PER = LV * 2
N_IN = G * G * PER
N_KEY = 6
CANDS = np.array([
    [1, 0, 0, 1, 0, 0],
    [1, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
], np.float32)
RNG = np.random.default_rng(0)


def encode(f: np.ndarray) -> np.ndarray:
    x = np.zeros(N_IN, np.float32)
    flat = (np.arange(G * G) * PER).ravel()
    lv = np.clip((f.ravel() * LV).astype(np.int32), 0, LV - 1)
    x[flat + lv] = 1.0
    x[flat + LV + (LV - 1 - lv)] = 1.0
    return x


def local(n_out, n_in, k):
    po = np.arange(n_out)[:, None] / max(1, n_out - 1)
    pi = np.arange(n_in)[None, :] / max(1, n_in - 1)
    idx = np.argsort(np.abs(po - pi), axis=1)[:, :k]
    M = np.zeros((n_out, n_in), np.float32)
    for o in range(n_out):
        M[o, idx[o]] = RNG.uniform(0.5, 1.5, k)
    return M


class BigPFC:
    def __init__(self, n2=768, n3=768, n4=512):
        self.F2 = local(n2, N_IN, 8)
        self.F3 = local(n3, n2, 8)
        self.F4 = local(n4, n3, 8)
        self.t2 = np.zeros(n2, np.float32)
        self.t3 = np.zeros(n3, np.float32)
        self.n4 = n4
        self.m = None
        self._cal = 0

    def hidden(self, X: np.ndarray) -> np.ndarray:
        d2 = X @ self.F2.T
        if self._cal < 2:
            self.t2[:] = np.quantile(d2, 0.7); self._cal += 1
        l2 = (d2 - self.t2 > 0).astype(np.float32)
        d3 = l2 @ self.F3.T
        if self._cal < 4:
            self.t3[:] = np.quantile(d3, 0.7); self._cal += 1
        l3 = (d3 - self.t3 > 0).astype(np.float32)
        d4 = l3 @ self.F4.T
        if self.m is None:
            self.m = max(1, int(X.sum(1).mean()))
        H = np.zeros_like(d4)
        for i in range(d4.shape[0]):
            H[i, np.argsort(d4[i])[-self.m:]] = 1.0
        return H


def render(w) -> np.ndarray:
    f = np.zeros((G, G), np.float32)
    cx = int(round(w.x))
    col = cx % G
    row = G - 1 - int(min(G - 1, round(w.y * 2)))
    f[max(0, row), col] = 1.0
    for c in range(G):
        if (cx - G // 2 + c) not in w.pits:
            f[G - 1, c] = max(f[G - 1, c], 0.4)
    return f


def new_pits(r):
    ps, x = [], 6
    while x < 60:
        ps.append((x, x + 2))
        x += int(r.integers(10, 15))
    return tuple(ps)
