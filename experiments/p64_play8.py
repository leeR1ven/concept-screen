"""8×8 + 死亡标签学习：环境已可玩、信息可见，看它能不能学会跳坑。"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.p51_screen2key import N_IN, PFC, encode  # noqa: E402
from experiments.p57_play import World  # noqa: E402

N_KEY = 6
CANDS = np.array([
    [1, 0, 0, 1, 0, 0],
    [1, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
], np.float32)


def main(ticks=8000, lr=0.4, explore=0.35, seed=0, horizon=2):
    pfc = PFC(N_IN)
    Wd = np.zeros((pfc.n4, 1), np.float32)
    u_h = np.zeros(pfc.n4, np.float32)
    u_d = np.zeros(1, np.float32)
    slots = np.arange(N_IN - N_KEY, N_IN)
    rng = np.random.default_rng(seed)
    w = World(seed=seed, pits=((6, 8), (20, 22), (34, 36)), length=60)
    best, eps, deaths, buf = 0.0, 0, [], []

    def learn(hb, danger):
        a = hb - u_h
        b = np.array([float(danger)], np.float32) - u_d
        Wd[:] += lr * np.outer(a, b) - 0.002 * Wd
        np.clip(Wd, -1.0, 1.0, out=Wd)

    for t in range(ticks):
        if not w.alive:
            deaths.append(round(float(w.x), 1))
            best = max(best, w.x)
            eps += 1
            for hb in buf:
                learn(hb, 1.0)
            buf.clear()
            w = World(seed=seed + eps, pits=((6, 8), (20, 22), (34, 36)), length=60)

        x = encode(w.frame())
        x[slots] = 0.0
        XA = np.repeat(x[None, :], len(CANDS), 0)
        XA[:, slots] = CANDS
        H = np.array([pfc.step(XA[i], learn=False) for i in range(len(CANDS))])
        if rng.random() < explore:
            act = int(rng.integers(len(CANDS)))
        else:
            risks = ((H - u_h) @ Wd)[:, 0] - u_d[0]
            act = int(np.argmin(risks))
        w.step(CANDS[act])
        h_now = H[act].copy()
        buf.append(h_now)
        if len(buf) > horizon:
            learn(buf.pop(0), 0.0)
        u_h += 0.005 * (h_now - u_h)
    best = max(best, w.x)
    hist = {}
    for d in deaths:
        hist[int(d // 10) * 10] = hist.get(int(d // 10) * 10, 0) + 1
    return dict(best=float(best), eps=eps + 1, hist=sorted(hist.items())[:6])


if __name__ == "__main__":
    for nm, kw in (("8x8_h2", dict()), ("8x8_explore5", dict(explore=0.5))):
        r = main(**kw)
        print("%-14s 最远 %.1f  局数 %d  死亡分布 %s"
              % (nm, r["best"], r["eps"], r["hist"]))
