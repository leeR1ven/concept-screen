"""新游戏：接东西。只能左右移动，接住从上方掉落的方块。

规则与平台跳跃完全不同：没有跳跃、没有坑，只有"横向对准"。
仍用同一套：8×8 编码 → 固定随机网络 → top-m → 时序赫布（学"这一步之后会漏吗"）→ 决策。
判据：接住率要明显超过随机（随机约 25%）。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.p51_screen2key import N_IN, PFC, encode  # noqa: E402

N_KEY = 6
G = 8
LEFT = np.array([0, 1, 0, 0, 0, 0], np.float32)
RIGHT = np.array([1, 0, 0, 0, 0, 0], np.float32)
CANDS = np.stack([LEFT, RIGHT])


class Catch:
    def __init__(self, rng):
        self.rng = rng
        self.reset()

    def reset(self):
        self.catcher = 3.0
        self.item_x = float(self.rng.integers(0, G))
        self.item_y = 0.0
        self.caught, self.missed = 0, 0

    def frame(self):
        f = np.zeros((G, G), np.float32)
        f[G - 1, int(round(self.catcher)) % G] = 1.0
        f[int(min(G - 1, round(self.item_y))), int(round(self.item_x)) % G] = 0.9
        return f

    def step(self, keys):
        if keys[1] > 0.5:
            self.catcher = max(0.0, self.catcher - 0.6)
        elif keys[0] > 0.5:
            self.catcher = min(G - 1.0, self.catcher + 0.6)
        self.item_y += 0.5
        if self.item_y >= G - 1:
            if abs(self.item_x - self.catcher) <= 0.8:
                self.caught += 1
            else:
                self.missed += 1
            self.item_x = float(self.rng.integers(0, G))
            self.item_y = 0.0


def run(policy, ticks=6000, lr=0.4, explore=0.4, seed=0):
    pfc = PFC(N_IN) if policy == "learn" else None
    Wd = np.zeros((pfc.n4, 1), np.float32) if pfc else None
    u_h = np.zeros(pfc.n4, np.float32) if pfc else None
    u_d = np.zeros(1, np.float32) if pfc else None
    slots = np.arange(N_IN - N_KEY, N_IN)
    rng = np.random.default_rng(seed)
    g = Catch(rng)
    buf = []
    prev_h = None

    step_box = [0]

    def learn(hb, danger, Wd=Wd):
        a = hb - u_h
        b = np.array([float(danger)], np.float32) - u_d
        # 凹形增长：越强越难涨
        phi = 1.0 / (1.0 + np.abs(Wd))
        Wd[:] += lr * np.outer(a, b) * phi
        np.clip(Wd, -1.0, 1.0, out=Wd)
        step_box[0] += 1
        if step_box[0] % 200 == 0:
            # 架构要求的统一降权：越弱越容易被降、被删
            Wd *= (1.0 - 0.05 * phi)
            Wd[np.abs(Wd) < 0.01] = 0.0

    for t in range(ticks):
        before = g.caught, g.missed
        x = encode(g.frame()); x[slots] = 0.0
        if policy == "random":
            keys = CANDS[int(rng.integers(2))]
            h_now = None
        else:
            XA = np.repeat(x[None, :], 2, 0)
            XA[:, slots] = CANDS
            H = np.array([pfc.step(XA[i], learn=False) for i in range(2)])
            if rng.random() < explore:
                act = int(rng.integers(2))
            else:
                act = int(np.argmin((((H - u_h) @ Wd) - u_d[0])[:, 0]))
            keys = CANDS[act]
            h_now = H[act]
        g.step(keys)
        if h_now is not None:
            near = g.item_y / (G - 1)
            off = min(1.0, abs(g.item_x - g.catcher) / 4.0)
            value = float(near * off)
            # 动作条件 + 稀疏化：只在"物品已进下半屏"时学（关键决策点）
            if prev_h is not None and near > 0.5:
                learn(prev_h, value)
            prev_h = h_now.copy()
            u_h += 0.005 * (h_now - u_h)
    return g.caught / max(1, g.caught + g.missed), g.caught, g.missed


if __name__ == "__main__":
    for name, kw in (("random", dict(policy="random")), ("learn", dict(policy="learn")),
                     ("learn_e60", dict(policy="learn", explore=0.6))):
        acc, c, m = run(**kw)
        print("%-10s 接住率 %.3f  （接住 %d / 漏 %d）" % (name, acc, c, m))
