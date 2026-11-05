"""世界 v2：坑 + 障碍块（要跳过去）+ 更长的关卡。检验"通关"泛化。

控制：2 个宏动作（前进 / 前进+跳）+ 情境门控（已验证有效的组合）。
判据：能走完全程（x 达到关卡长度）且过程中两类障碍都过。
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
A_RUN = np.array([1, 0, 0, 1, 0, 0], np.float32)
A_JUMP = np.array([1, 0, 1, 1, 0, 0], np.float32)
CANDS = np.stack([A_RUN, A_JUMP])


class World2:
    """坑（掉下去死）+ 障碍块（撞上死），都可跳过。"""

    def __init__(self, rng, length=120):
        self.length = length
        self.pits, self.blocks = set(), set()
        self.enemies = []
        x = 6
        while x < length - 6:
            if rng.random() < 0.5:
                self.pits |= {x, x + 1}
                x += int(rng.integers(10, 16))
            else:
                self.blocks.add(x)
                x += int(rng.integers(9, 14))
        self.reset()

    def spawn_enemies(self, n=6):
        self.enemies = []
        for i in range(n):
            x0 = 15 + i * (max(10, self.length // (n + 1)))
            if x0 >= self.length - 8:
                break
            self.enemies.append({"x": float(x0), "v": 0.25 if i % 2 else -0.25,
                                 "lo": float(x0 - 3), "hi": float(x0 + 3)})

    def reset(self):
        self.x, self.y, self.vy = 1.0, 0.0, 0.0
        self.alive, self.steps = True, 0

    def frame(self) -> np.ndarray:
        f = np.zeros((G, G), np.float32)
        cx = int(round(self.x))
        f[G - 1 - int(min(G - 1, round(self.y * 2))), cx % G] = 1.0
        for c in range(G):
            wx = cx - G // 2 + c
            col = c
            if wx not in self.pits:
                f[G - 1, col] = max(f[G - 1, col], 0.4)
            if wx in self.blocks:
                f[G - 2, col] = 1.0
            for e in self.enemies:
                if abs(e["x"] - wx) < 0.6:
                    f[G - 2, col] = 0.8
        return f

    def step(self, keys):
        if not self.alive:
            return
        self.steps += 1
        right, jump, run = keys[0] > .5, keys[2] > .5, keys[3] > .5
        vx = 0.5 if run else 0.25
        if right:
            self.x += vx
        if jump and self.y <= 0.05:
            self.vy = 1.35
        self.vy -= 0.25
        self.y = max(0.0, self.y + self.vy)
        if self.y <= 0:
            self.vy = 0.0
        cx = int(round(self.x))
        if cx in self.pits and self.y < 0.5:
            self.alive = False
        if cx in self.blocks and self.y < 1.0:
            self.alive = False
        # 移动敌人：碰到就死（跳起来躲得过去）
        for e in self.enemies:
            e["x"] += e["v"]
            if e["x"] < e["lo"] or e["x"] > e["hi"]:
                e["v"] = -e["v"]
            if abs(e["x"] - self.x) < 0.9 and self.y < 1.0:
                self.alive = False


def train(ticks=12000, lr=0.4, explore=0.35, gate=1.0, seed=0, length=120):
    pfc = PFC(N_IN)
    Wd = np.zeros((pfc.n4, 1), np.float32)
    u_h = np.zeros(pfc.n4, np.float32)
    u_d = np.zeros(1, np.float32)
    slots = np.arange(N_IN - N_KEY, N_IN)
    rng = np.random.default_rng(seed)
    wrng = np.random.default_rng(seed + 3)
    w = World2(wrng, length)
    w.spawn_enemies()
    best, eps, wins, deaths = 0.0, 0, 0, []
    buf = []

    def learn(hb, danger):
        a = hb - u_h
        b = np.array([float(danger)], np.float32) - u_d
        Wd[:] += lr * np.outer(a, b) - 0.002 * Wd
        np.clip(Wd, -1.0, 1.0, out=Wd)

    for t in range(ticks):
        if not w.alive or w.x >= w.length - 2:
            if w.x >= w.length - 2:
                wins += 1
            else:
                deaths.append(round(float(w.x), 1))
            best = max(best, w.x); eps += 1
            for hb in buf:
                learn(hb, 1.0 if w.x < w.length - 2 else 0.0)
            buf.clear()
            w = World2(wrng, length)
            w.spawn_enemies()
            if best > 20:                      # 课程学习：从离最远处 15 格的位置重开
                w.x = max(1.0, best - 15)

        x = encode(w.frame()); x[slots] = 0.0
        XA = np.repeat(x[None, :], 2, 0)
        XA[:, slots] = CANDS
        H = np.array([pfc.step(XA[i], learn=False) for i in range(2)])
        risks = (((H - u_h) @ Wd) - u_d[0])[:, 0]
        if rng.random() < explore:
            act = int(rng.integers(2))
        else:
            allowed = [0] if (gate > 0 and float(risks.min()) < gate) else [0, 1]
            act = min(allowed, key=lambda i: risks[i])
        w.step(CANDS[act])
        buf.append(H[act].copy())
        if len(buf) > 2:
            learn(buf.pop(0), 0.0)
        u_h += 0.005 * (H[act] - u_h)
    best = max(best, w.x)
    return dict(best=float(best), eps=eps, wins=wins, deaths=len(deaths))


if __name__ == "__main__":
    for L in (120,):
        r = train(length=L, ticks=25000)
        print("关卡长 %3d：最远 %6.1f  通关 %2d 次  局数 %3d"
              % (L, r["best"], r["wins"], r["eps"]))
