"""闭环 v3：直接学"这个动作之后会不会死"（死前两步回头补惩罚）。

复用 p60 的 16×16 编码与大前额叶。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.p57_play import World  # noqa: E402
from experiments.p62_core import (BigPFC, CANDS, N_IN, N_KEY, encode,  # noqa: E402
                                  new_pits, render)


def main(ticks=8000, lr=0.35, explore=0.35, seed=0, horizon=2):
    pfc = BigPFC()
    Wd = np.zeros((pfc.n4, 1), np.float32)
    u_h = np.zeros(pfc.n4, np.float32)
    u_d = np.zeros(1, np.float32)
    slots = np.arange(N_IN - N_KEY, N_IN)
    prng = np.random.default_rng(seed + 7)
    w = World(seed=seed, pits=new_pits(prng), length=70)
    rng = np.random.default_rng(seed)
    best, eps, deaths = 0.0, 0, []
    buf = []                     # 最近几步的 h（待判生死）

    def learn_one(h_vec, danger):
        a = h_vec - u_h
        b = np.array([float(danger)], np.float32) - u_d
        Wd[:] += lr * np.outer(a, b) - 0.002 * Wd
        np.clip(Wd, -1.0, 1.0, out=Wd)

    for t in range(ticks):
        if not w.alive:
            deaths.append(round(float(w.x), 1))
            best = max(best, w.x); eps += 1
            for hb in buf:                    # 死前 horizon 步 → 危险 1
                learn_one(hb, 1.0)
            buf.clear()
            w = World(seed=seed + eps, pits=new_pits(prng), length=70)

        x = encode(render(w)); x[slots] = 0.0
        XA = np.repeat(x[None, :], len(CANDS), 0)
        XA[:, slots] = CANDS
        H = pfc.hidden(XA)
        if rng.random() < explore:
            act = int(rng.integers(len(CANDS)))
        else:
            risks = ((H - u_h) @ Wd)[:, 0] - u_d[0]
            act = int(np.argmin(risks))
        w.step(CANDS[act])
        h_now = H[act].copy()
        buf.append(h_now)
        if len(buf) > horizon:
            old = buf.pop(0)
            learn_one(old, 0.0)               # 撑过 horizon 步没死 → 危险 0
        u_h += 0.005 * (h_now - u_h)
    best = max(best, w.x)
    hist = {}
    for d in deaths:
        hist[int(d // 10) * 10] = hist.get(int(d // 10) * 10, 0) + 1
    return dict(best=float(best), eps=eps + 1, deaths=len(deaths),
                hist=sorted(hist.items())[:6])


if __name__ == "__main__":
    for nm, kw in (("h2", dict(horizon=2)), ("h3", dict(horizon=3, explore=0.25))):
        r = main(**kw)
        print("%s 最远 %.1f  局数 %d  死亡分布 %s" % (nm, r["best"], r["eps"], r["hist"]))
