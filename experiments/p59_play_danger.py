"""闭环学习 v2：学"下一拍脚下会不会踩空"，据此选动作。

- 危险信号：下一拍画面里，角色所在列及其右一列的底部格子是否全暗（踩空）。
- 学习：h_t（含这一拍动作的副本）→ 危险标量，用时序赫布（凹形增长）。
- 决策：对每个候选动作算 h，预测危险；选危险最低的；危险高时优先跳。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.p51_screen2key import N_IN, PFC, encode  # noqa: E402
from experiments.p57_play import World  # noqa: E402

N_KEY = 6
CANDS = [
    ("right", np.array([1, 0, 0, 1, 0, 0], np.float32)),
    ("right_jump", np.array([1, 0, 1, 1, 0, 0], np.float32)),
    ("jump", np.array([0, 0, 1, 1, 0, 0], np.float32)),
    ("idle", np.zeros(N_KEY, np.float32)),
]


def bottom_cells(x: np.ndarray, col: int) -> np.ndarray:
    """第一层里"底部一行"对应的位置（每像素 8 个格，取正编码段的最低级）。"""
    base = (7 * 8 + col) * 8
    return np.arange(base, base + 8)


def ground_under(x: np.ndarray, col: int) -> float:
    """底部该列是否还有地面（正编码段的最低级亮着 = 地面在）。"""
    return float(x[bottom_cells(x, col)][0])


def main(ticks: int = 600, lr: float = 0.3, w_ref: float = 1.0, pw: float = 1.0,
         explore: float = 0.4, seed: int = 0):
    pfc = PFC(N_IN)
    Wd = np.zeros((pfc.n4, 1), np.float32)     # 最后一层 → 危险
    u_h = np.zeros(pfc.n4, np.float32)
    u_d = np.zeros(1, np.float32)
    slots = np.arange(N_IN - N_KEY, N_IN)
    def new_pits(r):
        ps, x = [], 7
        while x < 130:
            ps.append((x, x + 2))
            x += int(r.integers(12, 17))       # 间隔 12~16 格（跳一次能过）
        return tuple(ps)
    pit_rng = np.random.default_rng(seed + 999)
    w = World(seed=seed, pits=new_pits(pit_rng), length=140)
    rng = np.random.default_rng(seed)
    prev = None

    best_x, episodes = 0.0, 0
    deaths = []
    for t in range(ticks):
        if not w.alive:                       # 死了就重开（多局积累经验）
            deaths.append(round(float(w.x), 1))
            best_x = max(best_x, w.x)
            episodes += 1
            w = World(seed=seed + episodes, pits=new_pits(pit_rng), length=140)
            prev = None
        x = encode(w.frame())
        x[slots] = 0.0
        col = int(w.x) % 8

        if rng.random() < explore:
            act = int(rng.integers(len(CANDS)))
        else:
            risks = []
            for nm, keys in CANDS:
                xa = x.copy(); xa[slots] = keys
                h = pfc.step(xa, learn=False)
                risks.append(float(((h - u_h) @ Wd)[0] - u_d[0]))
            act = int(np.argmin(risks))        # 选预测危险最低的
        name, keys = CANDS[act]

        # 走一步，看真实结果
        w.step(keys)
        x_next = encode(w.frame())
        x_next[slots] = 0.0
        danger = 1.0 - ground_under(x_next, int(round(w.x)) % 8)

        xa = x.copy(); xa[slots] = keys
        h = pfc.step(xa, learn=True)
        if prev is not None:
            a = prev - u_h
            b = np.array([danger], np.float32) - u_d
            phi = 1.0 / (1.0 + np.abs(Wd) / w_ref) ** pw
            Wd += lr * np.outer(a, b) * phi - 0.002 * Wd * phi
            np.clip(Wd, -1.0, 1.0, out=Wd)
        u_h += 0.005 * (h - u_h)
        u_d += 0.005 * (np.array([danger], np.float32) - u_d)
        prev = h.copy()
    best_x = max(best_x, w.x)
    hist = {}
    for d in deaths:
        hist[int(d // 9) * 9] = hist.get(int(d // 9) * 9, 0) + 1
    return dict(x=float(best_x), steps=w.steps, alive=bool(w.alive),
                episodes=episodes + 1, deaths=deaths, hist=hist)


if __name__ == "__main__":
    print("%-10s %8s %8s %6s" % ("策略", "最远x", "步数", "活"))
    for nm, kw in (("learn", dict(ticks=20000)),
                   ("learn_fast", dict(ticks=20000, explore=0.2, lr=0.5))):
        r = main(**kw)
        print("%-10s 最远 %5.1f  局数 %4d  死亡位置分布 %s"
              % (nm, r["x"], r["episodes"],
                 sorted(r["hist"].items())[:8]))
