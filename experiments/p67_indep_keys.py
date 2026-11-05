"""独立按键控制：3 个键（右/B/跳）各自判断"按不按"，可同时按。

判据：随机关卡下最远 x > 500（宏动作 1612；8 组合 argmin 31.8）。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.p51_screen2key import N_IN, PFC, encode  # noqa: E402
from experiments.p57_play import World  # noqa: E402

N_KEY = 6
KEYS = [(0, "right"), (3, "B"), (2, "jump")]


def pits_fn(rng):
    ps, x = [], 6
    while x < 60:
        ps.append((x, x + 2)); x += int(rng.integers(10, 16))
    return tuple(ps)


def main(ticks=9000, lr=0.4, explore=0.3, seed=0, horizon=2):
    pfc = PFC(N_IN)
    Wd = np.zeros((pfc.n4, 1), np.float32)
    u_h = np.zeros(pfc.n4, np.float32)
    u_d = np.zeros(1, np.float32)
    slots = np.arange(N_IN - N_KEY, N_IN)
    rng = np.random.default_rng(seed)
    prng = np.random.default_rng(seed + 11)
    w = World(seed=seed, pits=pits_fn(prng), length=60)
    best, eps, deaths, buf = 0.0, 0, [], []
    thr = np.full(len(KEYS), 0.0, np.float32)      # 每个键自己的阈值（滑动均值）

    def learn(hb, danger):
        nonlocal Wd, u_d
        a = hb - u_h
        b = np.array([float(danger)], np.float32) - u_d
        Wd[:] += lr * np.outer(a, b) - 0.002 * Wd
        np.clip(Wd, -1.0, 1.0, out=Wd)

    for t in range(ticks):
        if not w.alive:
            deaths.append(round(float(w.x), 1))
            best = max(best, w.x); eps += 1
            for hb in buf:
                learn(hb, 1.0)
            buf.clear()
            w = World(seed=seed + eps, pits=pits_fn(prng), length=60)

        x = encode(w.frame()); x[slots] = 0.0
        keys = np.zeros(N_KEY, np.float32)
        hk = []
        for i, (idx, _) in enumerate(KEYS):
            xa = x.copy(); xa[slots[idx]] = 1.0
            h = pfc.step(xa, learn=False)
            hk.append(h)
            risk = float(((h - u_h) @ Wd)[0] - u_d[0])
            if rng.random() < explore:
                press = rng.random() < 0.5
            else:
                press = risk < thr[i]      # 各自独立过阈值
            keys[idx] = 1.0 if press else 0.0
            thr[i] += 0.01 * (risk - thr[i])
        w.step(keys)
        # 用"合起来的按键"那一拍的 h 做学习
        xa = x.copy(); xa[slots] = keys
        h_now = pfc.step(xa, learn=True)
        buf.append(h_now)
        if len(buf) > horizon:
            learn(buf.pop(0), 0.0)
        u_h += 0.005 * (h_now - u_h)
    best = max(best, w.x)
    hist = {}
    for d in deaths:
        hist[int(d // 10) * 10] = hist.get(int(d // 10) * 10, 0) + 1
    return dict(best=float(best), eps=eps + 1, hist=sorted(hist.items())[:5])


if __name__ == "__main__":
    for nm, kw in (("indep", dict()), ("indep_e50", dict(explore=0.5))):
        r = main(**kw)
        print("%-10s 最远 %8.1f  局数 %d  死亡 %s"
              % (nm, r["best"], r["eps"], r["hist"]))
