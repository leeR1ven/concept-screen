"""闭环游戏世界：模型按键直接驱动角色，用"走了多远/活多久"打分。

世界：地面 + 坑（掉下去算死）。按键：right 前进、B 加速、A 跳。
没有老师标签——只看它玩得怎么样。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.p51_screen2key import N_IN, PFC, encode  # noqa: E402

W, H = 8, 8
KEYS = ("right", "left", "A", "B", "down", "up")


class World:
    def __init__(self, seed=0, pits=((6, 8), (20, 22), (34, 36)), length=60):
        self.rng = np.random.default_rng(seed)
        self.pits = set()
        for a, b in pits:
            self.pits.update(range(a, b))
        self.length = length
        self.reset()

    def reset(self):
        self.x = 1.0
        self.y = 0.0          # 0 = 地面
        self.vy = 0.0
        self.alive = True
        self.steps = 0
        self.t = 0

    def frame(self) -> np.ndarray:
        f = np.zeros((H, W), np.float32)
        col = int(self.x) % W
        row = H - 1 - int(min(3, round(self.y)))
        f[row, col] = 1.0
        # 地面（坑的位置不画）
        for c in range(W):
            wx = int(self.x) - 2 + c
            if wx not in self.pits:
                f[H - 1, c] = max(f[H - 1, c], 0.4)
        return f

    def step(self, keys: np.ndarray):
        if not self.alive:
            return
        self.steps += 1
        right = keys[0] > 0.5
        left = keys[1] > 0.5
        jump = keys[2] > 0.5
        run = keys[3] > 0.5
        vx = (0.5 if run else 0.25)
        if right:
            self.x += vx
        elif left:
            self.x = max(0.0, self.x - vx * 0.5)
        # 跳跃物理
        if jump and self.y <= 0.05:
            self.vy = 1.35
        self.vy -= 0.25
        self.y = max(0.0, self.y + self.vy)
        if self.y <= 0.0:
            self.vy = 0.0
        # 掉坑
        if int(round(self.x)) in self.pits and self.y < 0.5:
            self.alive = False
        self.t += 1


def run_policy(kind: str, seed: int = 0, ticks: int = 600, model=None):
    w = World(seed=seed)
    pfc = model["pfc"] if model else None
    Wb = model["Wb"] if model else None
    u_h = model["u_h"] if model else None
    u_b = model["u_b"] if model else None
    for t in range(ticks):
        if kind == "random":
            keys = (np.random.default_rng(t).random(6) > 0.5).astype(np.float32)
        elif kind == "always_right":
            keys = np.zeros(6, np.float32); keys[0] = 1.0; keys[3] = 1.0
        elif kind == "oracle":
            # 会玩的人：一直右+B，坑前一格跳
            keys = np.zeros(6, np.float32); keys[0] = 1.0; keys[3] = 1.0
            nxt = int(w.x) + 1
            if nxt in w.pits and w.y <= 0.05:
                keys[2] = 1.0
        elif kind == "model":
            f = w.frame()
            h = pfc.step(encode(f), learn=False)
            drive = (h - u_h) @ Wb
            keys = (drive > 0).astype(np.float32)
        else:
            raise ValueError(kind)
        w.step(keys)
        if not w.alive:
            break
    return dict(x=float(w.x), steps=w.steps, alive=bool(w.alive))


def main():
    print("%-12s %8s %8s %6s" % ("策略", "最远x", "步数", "活"))
    for kind in ("always_right", "random", "oracle"):
        r = run_policy(kind)
        print("%-12s %8.1f %8d %6s" % (kind, r["x"], r["steps"], r["alive"]))


if __name__ == "__main__":
    main()
