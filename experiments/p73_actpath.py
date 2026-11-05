"""动作专用通路：最后一层的推动 = 视觉推动 + A·（按键副本）。

A 是出生固定的随机矩阵（模拟"意图副本 → 最后一层"的通路），
目的是让 6 格按键信号不被 128 格视觉淹没（这是上一轮测到"跳/不跳风险完全相同"的原因）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.p51_screen2key import N_IN, PFC, encode  # noqa: E402
from experiments.p68_world2 import CANDS  # noqa: E402
from experiments.p71_ladder import Level1  # noqa: E402

N_KEY = 6
RNG = np.random.default_rng(7)


class PFC2(PFC):
    """在最后一层给动作留一条强通路。"""

    def __init__(self, n1):
        super().__init__(n1)
        self.A = RNG.normal(0, 1.0, (self.n4, N_KEY)).astype(np.float32) * 2.5

    def step(self, x, learn=True):
        act6 = np.asarray(x[-N_KEY:], np.float32)
        base = super().step(x, learn=learn)
        d = (self.F4 @ (self._l3 if hasattr(self, "_l3") else np.zeros(self.F4.shape[1],
                                                                     np.float32))
             + self.A @ act6)
        h = np.zeros(self.n4, np.float32)
        h[np.argsort(d)[-self.m:]] = 1.0
        return h


def train(ticks=8000, lr=0.4, explore=0.35, seed=0, length=40):
    pfc = PFC2(N_IN)
    Wd = np.zeros((pfc.n4, 1), np.float32)
    u_h = np.zeros(pfc.n4, np.float32)
    u_d = np.zeros(1, np.float32)
    slots = np.arange(N_IN - N_KEY, N_IN)
    rng = np.random.default_rng(seed)
    wrng = np.random.default_rng(seed + 3)
    w = Level1(wrng, length)
    buf, eps, wins = [], 0, 0
    n_jump, n_step = 0, 0
    for t in range(ticks):
        if not w.alive or w.x >= w.length - 2:
            wins += 1 if w.x >= w.length - 2 else 0
            eps += 1
            if not w.alive:
                # 只标"坑前没跳"的步；跳过的步标安全（修正因果方向）
                pits = sorted(w.pits)
                for hb, xb, jumped in buf:
                    ahead = min([p - xb for p in pits if p > xb] or [99])
                    if ahead <= 3:
                        danger = 1.0 if not jumped else 0.0   # 坑前：该跳
                    else:
                        danger = 0.5 if jumped else 0.0       # 安全地段：不该跳
                    a = hb - u_h
                    b = np.array([danger], np.float32) - u_d
                    Wd[:] += lr * np.outer(a, b) - 0.002 * Wd
                    np.clip(Wd, -1.0, 1.0, out=Wd)
            buf.clear()
            w = Level1(wrng, length)
        x = encode(w.frame()); x[slots] = 0.0
        XA = np.repeat(x[None, :], 2, 0)
        XA[:, slots] = CANDS
        H = np.array([pfc.step(XA[i], learn=False) for i in range(2)])
        risks = (((H - u_h) @ Wd) - u_d[0])[:, 0] / max(1, pfc.m)
        act = int(rng.integers(2)) if rng.random() < explore else int(np.argmin(risks))
        x_before = float(w.x)
        w.step(CANDS[act])
        n_jump += 1 if CANDS[act][2] > 0.5 else 0
        n_step += 1
        buf.append((H[act].copy(), x_before, bool(CANDS[act][2] > 0.5)))
        if len(buf) > 2:
            a = buf.pop(0)[0] - u_h
            b = np.array([0.0], np.float32) - u_d
            Wd[:] += lr * np.outer(a, b) - 0.002 * Wd
            np.clip(Wd, -1.0, 1.0, out=Wd)
        u_h += 0.005 * (H[act] - u_h)
    # 探测：距坑 2 格
    w = Level1(np.random.default_rng(0), length); w.reset()
    w.pits = {6, 7}; w.x = 4.0
    x = encode(w.frame()); x[slots] = 0.0
    XA = np.repeat(x[None, :], 2, 0); XA[:, slots] = CANDS
    H = np.array([pfc.step(XA[i], learn=False) for i in range(2)])
    r = (((H - u_h) @ Wd) - u_d[0])[:, 0] / max(1, pfc.m)
    return wins / max(1, eps), eps, float(r[0]), float(r[1]), n_jump / max(1, n_step)


if __name__ == "__main__":
    tr, eps, r0, r1, jr = train()
    print("训练通关率 %.2f（%d 局）" % (tr, eps))
    print("距坑2格：风险[不跳]=%+.4f 风险[跳]=%+.4f 差=%+.4f  跳的比例 %.2f"
          % (r0, r1, r1 - r0, jr))
