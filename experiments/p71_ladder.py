"""难度阶梯：Level 1 = 40 格、只有坑、每局随机布局、无课程重开。

判据：训练通关率 > 50%，且没见过的关卡也 > 30%（才算学到规则）。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.p51_screen2key import N_IN, PFC, encode  # noqa: E402
from experiments.p68_world2 import CANDS, World2  # noqa: E402

N_KEY = 6


class Level1(World2):
    """只有坑，没有障碍块和敌人。"""

    def __init__(self, rng, length=40):
        self.length = length
        self.pits, self.blocks, self.enemies = set(), set(), []
        x = 6
        while x < length - 5:
            self.pits |= {x, x + 1}
            x += int(rng.integers(9, 14))
        self.reset()

    def spawn_enemies(self, n=0):
        self.enemies = []


def run(ticks=20000, lr=0.4, explore=0.35, gate=1.0, seed=0, length=40):
    pfc = PFC(N_IN)
    Wd = np.zeros((pfc.n4, 1), np.float32)
    u_h = np.zeros(pfc.n4, np.float32)
    u_d = np.zeros(1, np.float32)
    slots = np.arange(N_IN - N_KEY, N_IN)
    rng = np.random.default_rng(seed)
    wrng = np.random.default_rng(seed + 3)
    w = Level1(wrng, length)
    eps = wins = 0
    buf = []
    y = np.zeros(N_IN, np.float32)      # 上一拍的意图（保持）
    for t in range(ticks):
        if not w.alive or w.x >= w.length - 2:
            survived = w.x >= w.length - 2
            wins += 1 if survived else 0
            eps += 1
            # 关键：死了就把最近两步标成"危险"，否则模型永远学不到要跳
            if not survived:
                for hb in buf:
                    a = hb - u_h
                    b = np.array([1.0], np.float32) - u_d
                    Wd[:] += lr * 5.0 * np.outer(a, b) - 0.002 * Wd   # 死亡样本加权
                    np.clip(Wd, -1.0, 1.0, out=Wd)
            buf.clear()
            w = Level1(wrng, length)
        x = encode(w.frame()); x[slots] = 0.0
        x = np.maximum(x, y)                 # 第一层 = 画面 ∨ 意图（保持住上一拍的动作）
        XA = np.repeat(x[None, :], 2, 0)
        XA[:, slots] = CANDS
        H = np.array([pfc.step(XA[i], learn=False) for i in range(2)])
        risks = (((H - u_h) @ Wd) - u_d[0])[:, 0]
        act = int(rng.integers(2)) if rng.random() < explore else \
            min([0] if (gate > 0 and risks.min() < gate) else [0, 1], key=lambda i: risks[i])
        w.step(CANDS[act])
        y[slots] = CANDS[act] * 0.7          # 意图衰减保持（模拟回路）
        buf.append(H[act].copy())
        if len(buf) > 2:
            a = buf.pop(0) - u_h
            b = np.array([0.0], np.float32) - u_d
            Wd[:] += lr * np.outer(a, b) - 0.002 * Wd
            np.clip(Wd, -1.0, 1.0, out=Wd)
        u_h += 0.005 * (H[act] - u_h)

    # 考试：没见过的布局，冻结权重
    ev = np.random.default_rng(seed + 999)
    wins_ev = trials = 0
    for _ in range(40):
        w = Level1(ev, length)
        while w.alive and w.x < w.length - 2 and w.steps < 2000:
            x = encode(w.frame()); x[slots] = 0.0
            XA = np.repeat(x[None, :], 2, 0)
            XA[:, slots] = CANDS
            H = np.array([pfc.step(XA[k], learn=False) for k in range(2)])
            risks = (((H - u_h) @ Wd) - u_d[0])[:, 0]
            act = min([0] if (gate > 0 and risks.min() < gate) else [0, 1],
                      key=lambda k: risks[k])
            w.step(CANDS[act])
        wins_ev += 1 if w.x >= w.length - 2 else 0
        trials += 1
    return wins / max(1, eps), eps, wins_ev / max(1, trials), trials


if __name__ == "__main__":
    tr, eps, ev, n = run()
    print("Level1（40格/只坑/随机布局）训练通关率 %.2f（%d 局） 新关卡通关率 %.2f（%d 局）"
          % (tr, eps, ev, n))
