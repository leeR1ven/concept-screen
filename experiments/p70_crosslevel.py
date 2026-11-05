"""跨关卡泛化：训练用 A/B 两关，测试用没见过的 C 关。

判据：C 关通关率要明显高于 0（且接近训练关水平），才算"学会这个游戏"而不是记住一关。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.p51_screen2key import N_IN, PFC, encode  # noqa: E402
from experiments.p68_world2 import CANDS, World2  # noqa: E402


def build(rng, length=120):
    w = World2(rng, length)
    w.spawn_enemies()
    return w


def train(ticks=25000, lr=0.4, explore=0.35, gate=1.0, seed=0, length=120):
    pfc = PFC(N_IN)
    Wd = np.zeros((pfc.n4, 1), np.float32)
    u_h = np.zeros(pfc.n4, np.float32)
    u_d = np.zeros(1, np.float32)
    slots = np.arange(N_IN - 6, N_IN)
    rng = np.random.default_rng(seed)
    wrng = np.random.default_rng(seed + 3)
    w = build(wrng, length)
    best, eps, wins, buf = 0.0, 0, 0, []

    for t in range(ticks):
        if not w.alive or w.x >= w.length - 2:
            if w.x >= w.length - 2:
                wins += 1
            best = max(best, w.x); eps += 1
            w = build(wrng, length)      # 每局全新布局，不做课程重开
        x = encode(w.frame()); x[slots] = 0.0
        XA = np.repeat(x[None, :], 2, 0)
        XA[:, slots] = CANDS
        H = np.array([pfc.step(XA[i], learn=False) for i in range(2)])
        risks = (((H - u_h) @ Wd) - u_d[0])[:, 0]
        act = int(rng.integers(2)) if rng.random() < explore else \
            min([0] if (gate > 0 and risks.min() < gate) else [0, 1], key=lambda i: risks[i])
        w.step(CANDS[act])
        buf.append(H[act].copy())
        if len(buf) > 2:
            a = buf.pop(0) - u_h
            b = np.array([0.0], np.float32) - u_d
            Wd[:] += lr * np.outer(a, b) - 0.002 * Wd
            np.clip(Wd, -1.0, 1.0, out=Wd)
        u_h += 0.005 * (H[act] - u_h)
    return pfc, Wd, u_h, u_d, wins, eps


def evaluate(pfc, Wd, u_h, u_d, seed=999, levels=30, length=120, gate=1.0):
    rng = np.random.default_rng(seed)
    slots = np.arange(N_IN - 6, N_IN)
    wins, trials = 0, 0
    for i in range(levels):
        w = build(rng, length)
        while w.alive and w.x < w.length - 2 and w.steps < 3000:
            x = encode(w.frame()); x[slots] = 0.0
            XA = np.repeat(x[None, :], 2, 0)
            XA[:, slots] = CANDS
            H = np.array([pfc.step(XA[k], learn=False) for k in range(2)])
            risks = (((H - u_h) @ Wd) - u_d[0])[:, 0]
            act = min([0] if (gate > 0 and risks.min() < gate) else [0, 1],
                      key=lambda k: risks[k])
            w.step(CANDS[act])
        wins += 1 if w.x >= w.length - 2 else 0
        trials += 1
    return wins / max(1, trials)


if __name__ == "__main__":
    pfc, Wd, u_h, u_d, wins, eps = train()
    print("训练：通关 %d / %d 局（%.0f%%）" % (wins, eps, 100 * wins / max(1, eps)))
    print("同一分布的新关卡 通关率: %.2f" % evaluate(pfc, Wd, u_h, u_d, seed=999))
