"""闭环学习：学"我这一拍 → 下一拍画面"，再用预测来选动作（避免掉坑）。

每一步：
  1. 对每个候选动作，把"按键副本"放进第一层尾部，算 h，再用 Wnext 预测下一帧；
  2. 选"预测画面里脚下还有地面"的那个动作；
  3. 真正走一步，用真实下一帧更新 Wnext（协方差 + 凹形增长）。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.p51_screen2key import N_IN, PFC, encode  # noqa: E402
from experiments.p57_play import World  # noqa: E402

N_KEY = 6
CANDS = [
    ("right", np.array([1, 0, 0, 1, 0, 0], np.float32)),   # 右 + B
    ("right_jump", np.array([1, 0, 1, 1, 0, 0], np.float32)),
    ("jump", np.array([0, 0, 1, 1, 0, 0], np.float32)),
    ("idle", np.zeros(N_KEY, np.float32)),
]


def main(ticks: int = 600, lr: float = 0.2, w_ref: float = 1.0, pw: float = 1.0,
         warm: int = 120, seed: int = 0, verbose: bool = False):
    pfc = PFC(N_IN)
    Wnext = np.zeros((pfc.n4, N_IN), np.float32)
    u_h = np.zeros(pfc.n4, np.float32)
    u_x = np.zeros(N_IN, np.float32)
    motor_slots = np.arange(N_IN - N_KEY, N_IN)     # 第一层尾部放按键副本
    w = World(seed=seed)
    prev = None                                      # (h_with_action, x_next)
    rng = np.random.default_rng(seed)

    for t in range(ticks):
        if not w.alive:
            break
        f = w.frame()
        x = encode(f)
        x[motor_slots] = 0.0

        if t < warm or rng.random() < 0.15:
            act = int(rng.integers(len(CANDS)))      # 前期随便试，探索
            name, keys = CANDS[act]
        else:
            best, best_score = 0, -1e9
            for i, (nm, keys) in enumerate(CANDS):
                xa = x.copy(); xa[motor_slots] = keys
                h = pfc.step(xa, learn=False)
                pred = h @ Wnext - u_x                # 预测下一拍输入
                # 只看"下一拍脚下有没有地面"（第一层底部 + 当前列附近）
                col = int(w.x) % 8
                idx = [(7 * 8 + c) * 8 for c in range(8)]   # 底部一行的像素块起点
                score = float(pred[idx].mean())
                if score > best_score:
                    best, best_score = i, score
            act = best
            name, keys = CANDS[act]

        # 学习：上一拍的 h（带上一拍的动作）→ 这一拍真实画面
        if prev is not None:
            h_prev, x_prev = prev
            a = h_prev - u_h
            tgt = x - u_x
            phi = 1.0 / (1.0 + np.abs(Wnext) / w_ref) ** pw
            Wnext += lr * np.outer(a, tgt) * phi - 0.002 * Wnext * phi
            np.clip(Wnext, -1.0, 1.0, out=Wnext)

        xa = x.copy(); xa[motor_slots] = keys
        h = pfc.step(xa, learn=True)
        prev = (h.copy(), x.copy())
        u_h += 0.005 * (h - u_h)
        u_x += 0.005 * (x - u_x)
        w.step(keys)
        if verbose and t % 50 == 0:
            print("t=%3d x=%.1f act=%s" % (t, w.x, name))
    return dict(x=float(w.x), steps=w.steps, alive=bool(w.alive))


if __name__ == "__main__":
    print("%-10s %8s %8s %6s" % ("策略", "最远x", "步数", "活"))
    for nm, kw in (("learn", {}),):
        r = main(**kw)
        print("%-10s %8.1f %8d %6s" % (nm, r["x"], r["steps"], r["alive"]))
