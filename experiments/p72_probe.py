"""决定性测量：距坑 2 格时，跳 vs 不跳 的预测风险。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.p51_screen2key import N_IN, PFC, encode  # noqa: E402
from experiments.p68_world2 import CANDS  # noqa: E402
from experiments.p71_ladder import Level1  # noqa: E402

N_KEY = 6


def train_brief(ticks=8000, lr=0.4, explore=0.35, seed=0, length=40):
    pfc = PFC(N_IN)
    Wd = np.zeros((pfc.n4, 1), np.float32)
    u_h = np.zeros(pfc.n4, np.float32)
    u_d = np.zeros(1, np.float32)
    slots = np.arange(N_IN - N_KEY, N_IN)
    rng = np.random.default_rng(seed)
    wrng = np.random.default_rng(seed + 3)
    w = Level1(wrng, length)
    buf = []
    for t in range(ticks):
        if not w.alive or w.x >= w.length - 2:
            if not w.alive:
                for hb in buf:
                    a = hb - u_h
                    b = np.array([1.0], np.float32) - u_d
                    Wd[:] += lr * np.outer(a, b) - 0.002 * Wd
                    np.clip(Wd, -1.0, 1.0, out=Wd)
            buf.clear()
            w = Level1(wrng, length)
        x = encode(w.frame()); x[slots] = 0.0
        XA = np.repeat(x[None, :], 2, 0)
        XA[:, slots] = CANDS
        H = np.array([pfc.step(XA[i], learn=False) for i in range(2)])
        risks = (((H - u_h) @ Wd) - u_d[0])[:, 0]
        act = int(rng.integers(2)) if rng.random() < explore else int(np.argmin(risks))
        w.step(CANDS[act])
        buf.append(H[act].copy())
        if len(buf) > 2:
            a = buf.pop(0) - u_h
            b = np.array([0.0], np.float32) - u_d
            Wd[:] += lr * np.outer(a, b) - 0.002 * Wd
            np.clip(Wd, -1.0, 1.0, out=Wd)
        u_h += 0.005 * (H[act] - u_h)
    return pfc, Wd, u_h, u_d


def probe(pfc, Wd, u_h, u_d):
    slots = np.arange(N_IN - N_KEY, N_IN)
    w = Level1(np.random.default_rng(0), 40)
    w.reset()
    w.pits = {6, 7}          # 固定：坑在 6
    w.x = 4.0                # 距坑 2 格
    x = encode(w.frame()); x[slots] = 0.0
    XA = np.repeat(x[None, :], 2, 0)
    XA[:, slots] = CANDS
    H = np.array([pfc.step(XA[i], learn=False) for i in range(2)])
    r = (((H - u_h) @ Wd) - u_d[0])[:, 0]
    print("距坑2格：风险[不跳]=%+.4f  风险[跳]=%+.4f  差=%+.4f"
          % (r[0], r[1], r[1] - r[0]))
    print("=> 跳更低才可能学会；若跳更高，argmin 永远不会跳")


if __name__ == "__main__":
    pfc, Wd, u_h, u_d = train_brief()
    probe(pfc, Wd, u_h, u_d)
