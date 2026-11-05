"""放大版：16×16 画面、更大的前额叶，闭环学"别踩空"。

编码：每像素 4 级温度计正反对（8 格），16×16 → 2048 个输入。
候选动作的 h 一次矩阵乘算完（批量），避免 4 倍开销。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.p57_play import World  # noqa: E402

G = 16                    # 画面边长
LV = 4                    # 每像素等级
PER = LV * 2              # 每像素占用格数（正反）
N_IN = G * G * PER
N_KEY = 6
CANDS = np.array([
    [1, 0, 0, 1, 0, 0],    # right+B
    [1, 0, 1, 1, 0, 0],    # right+B+jump
    [0, 0, 1, 1, 0, 0],    # jump
    [0, 0, 0, 0, 0, 0],    # idle
], np.float32)
RNG = np.random.default_rng(0)


def encode(f: np.ndarray) -> np.ndarray:
    x = np.zeros(N_IN, np.float32)
    idx = (np.arange(G * G) * PER).reshape(G, G)
    flat = idx.ravel()
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
        """X: (batch, N_IN) → 最后一层 (batch, n4)。"""
        l2 = (X @ self.F2.T - self.t2 > 0).astype(np.float32)
        if self._cal < 2:
            self.t2[:] = np.quantile(X @ self.F2.T, 0.7)
            self._cal += 1
        l3 = (l2 @ self.F3.T - self.t3 > 0).astype(np.float32)
        if self._cal < 4:
            self.t3[:] = np.quantile(l2 @ self.F3.T, 0.7)
            self._cal += 1
        d4 = l3 @ self.F4.T
        if self.m is None:
            self.m = max(1, int(X.sum(1).mean()))
        H = np.zeros_like(d4)
        for i in range(d4.shape[0]):
            H[i, np.argsort(d4[i])[-self.m:]] = 1.0
        return H


def render(w) -> np.ndarray:
    """按 G×G 分辨率画出当前世界：角色 + 地面（坑处不画）。"""
    f = np.zeros((G, G), np.float32)
    cx = int(round(w.x))
    col = cx % G
    row = G - 1 - int(min(G - 1, round(w.y * 2)))
    f[max(0, row), col] = 1.0
    for c in range(G):
        wx = cx - G // 2 + c
        if wx not in w.pits:
            f[G - 1, c] = max(f[G - 1, c], 0.4)
    return f


def new_pits(r):
    ps, x = [], 6
    while x < 60:
        ps.append((x, x + 2))
        x += int(r.integers(10, 15))
    return tuple(ps)


def main(ticks=6000, lr=0.3, explore=0.35, seed=0):
    pfc = BigPFC()
    Wd = np.zeros((pfc.n4, 1), np.float32)
    u_h = np.zeros(pfc.n4, np.float32)
    u_d = np.zeros(1, np.float32)
    slots = np.arange(N_IN - N_KEY, N_IN)
    prng = np.random.default_rng(seed + 7)
    w = World(seed=seed, pits=new_pits(prng), length=70)
    rng = np.random.default_rng(seed)
    prev_h, best, eps, deaths = None, 0.0, 0, []
    buf = []

    for t in range(ticks):
        if not w.alive:
            deaths.append(round(float(w.x), 1))
            best = max(best, w.x); eps += 1
            w = World(seed=seed + eps, pits=new_pits(prng), length=70)
            prev_h = None
        x = encode(render(w)); x[slots] = 0.0
        XA = np.repeat(x[None, :], len(CANDS), 0)
        XA[:, slots] = CANDS
        if rng.random() < explore:
            act = int(rng.integers(len(CANDS)))
        else:
            H = pfc.hidden(XA)
            risks = (H - u_h) @ Wd - u_d[0]
            act = int(np.argmin(risks))
        w.step(CANDS[act])
        alive = bool(w.alive)
        # 教学信号：把最近两步的 (状态, 动作) 记下来；
        # 死了 → 这两步都标"危险 1"；活着走满两步 → 标"危险 0"。
        buf.append(H[act].copy())
        if len(buf) > 2:
            buf.pop(0)
        if not alive:
            for hb in buf:
                a_ = hb - u_h
                b_ = np.array([1.0], np.float32) - u_d
                Wd += lr * np.outer(a_, b_) - 0.002 * Wd
                np.clip(Wd, -1.0, 1.0, out=Wd)
            buf.clear()
        elif len(buf) == 2 and t % 1 == 0:
            hb = buf[0]
            a_ = hb - u_h
            b_ = np.array([0.0], np.float32) - u_d
            Wd += lr * np.outer(a_, b_) - 0.002 * Wd
            np.clip(Wd, -1.0, 1.0, out=Wd)
            buf.pop(0)
        H = pfc.hidden(XA)
            risks = (H - u_h) @ Wd - u_d[0]
            act = int(np.argmin(risks))
        w.step(CANDS[act])
        x_next = encode(render(w)); x_next[slots] = 0.0
        # 前瞻两格：接下来 1~2 格内脚下会不会空（提前预警，才来得及跳）
        cx = int(round(w.x))
        ahead = [1.0 - float(x_next[(G - 1) * G * PER + ((cx + c) % G) * PER])
                 for c in (1, 2)]
        danger = float(max(ahead))
        H = pfc.hidden(XA) if True else H
        prev_h = H[act].copy()
    best = max(best, w.x)
    hist = {}
    for d in deaths:
        hist[int(d // 10) * 10] = hist.get(int(d // 10) * 10, 0) + 1
    return dict(best=float(best), eps=eps + 1, hist=sorted(hist.items())[:6])


if __name__ == "__main__":
    r = main()
    print("16×16 模型：最远 %.1f  局数 %d  死亡分布 %s" % (r["best"], r["eps"], r["hist"]))
