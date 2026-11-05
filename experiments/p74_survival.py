"""改成"预测还能活多少步"（存活时间价值），替代"下一步危险"。

理由：下一步危险里，"跳起来"看起来总是安全的 → 连跳；而"存活时间"里连跳会立刻缩短寿命。
学习：每局结束（死或通关）后，对该局的每个状态用实际剩余步数做目标（蒙特卡洛 + 赫布）。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.p51_screen2key import N_IN, PFC, encode  # noqa: E402
from experiments.p68_world2 import CANDS  # noqa: E402
from experiments.p71_ladder import Level1  # noqa: E402

N_KEY = 6


def run(ticks=20000, lr=0.05, explore=0.3, seed=0, length=40, horizon=60):
    pfc = PFC(N_IN)
    Wv = np.zeros((pfc.n4, 1), np.float32)     # 最后一层 → 存活步数
    u_h = np.zeros(pfc.n4, np.float32)
    u_v = np.zeros(1, np.float32)
    slots = np.arange(N_IN - N_KEY, N_IN)
    rng = np.random.default_rng(seed)
    wrng = np.random.default_rng(seed + 3)
    w = Level1(wrng, length)
    traj = []                                   # [(h, x)] 本局
    eps = wins = 0

    def learn_episode(traj, final_score):
        # 对每步用"实际剩余步数 / horizon"当目标
        n = len(traj)
        for i, h in enumerate(traj):
            remaining = (n - i) if final_score < 1.0 else (horizon - i)
            target = float(np.clip(remaining / horizon, 0.0, 1.0))
            a = h - u_h
            b = np.array([target], np.float32) - u_v
            Wv[:] += lr * np.outer(a, b)
            np.clip(Wv, -1.0, 1.0, out=Wv)

    for t in range(ticks):
        if not w.alive or w.x >= w.length - 2:
            finished = w.x >= w.length - 2
            wins += 1 if finished else 0
            eps += 1
            learn_episode(traj, 1.0 if finished else 0.0)
            traj.clear()
            w = Level1(wrng, length)

        x = encode(w.frame()); x[slots] = 0.0
        XA = np.repeat(x[None, :], 2, 0)
        XA[:, slots] = CANDS
        H = np.array([pfc.step(XA[i], learn=False) for i in range(2)])
        vals = (((H - u_h) @ Wv) - u_v[0])[:, 0]
        act = int(rng.integers(2)) if rng.random() < explore else int(np.argmax(vals))
        w.step(CANDS[act])
        traj.append(H[act].copy())
        u_h += 0.005 * (H[act] - u_h)

    # 考试：没见过的布局
    ev = np.random.default_rng(seed + 999)
    wins_ev = trials = 0
    for _ in range(40):
        w = Level1(ev, length)
        while w.alive and w.x < w.length - 2 and w.steps < 2000:
            x = encode(w.frame()); x[slots] = 0.0
            XA = np.repeat(x[None, :], 2, 0)
            XA[:, slots] = CANDS
            H = np.array([pfc.step(XA[k], learn=False) for k in range(2)])
            vals = (((H - u_h) @ Wv) - u_v[0])[:, 0]
            w.step(CANDS[int(np.argmax(vals))])
        wins_ev += 1 if w.x >= w.length - 2 else 0
        trials += 1
    return wins / max(1, eps), eps, wins_ev / max(1, trials), trials


if __name__ == "__main__":
    tr, e, ev, n = run()
    print("存活时间价值：训练通关 %.2f（%d 局） 新关卡 %.2f（%d 局）" % (tr, e, ev, n))
