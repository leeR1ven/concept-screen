"""排查：①物理上"提前跳"到底能不能过；②最后一层能不能区分"前面有坑"；
③分辨率的影响；④教学信号的时间对齐。"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.p57_play import World  # noqa: E402
from experiments.p62_core import BigPFC, encode, new_pits, render  # noqa: E402


def test_physics():
    """脚本策略：距坑 k 格时起跳，看能不能过。"""
    rng = np.random.default_rng(0)
    res = {}
    for k in (1, 2, 3, 4):
        survived, trials = 0, 30
        for s in range(trials):
            w = World(seed=s, pits=new_pits(rng), length=50)
            for t in range(400):
                if not w.alive:
                    break
                nxt = int(round(w.x)) + 1
                jump = any((nxt + i) in w.pits for i in range(k))
                keys = np.array([1, 0, 1 if jump else 0, 1, 0, 0], np.float32)
                w.step(keys)
            survived += 1 if w.alive else 0
        res[k] = survived / trials
    print("① 物理：距坑 k 格起跳的存活率:", {k: round(v, 2) for k, v in res.items()})
    return res


def test_representation():
    """最后一层能否区分"前方 2 格内有坑"——用线性探针（只做诊断，不参与决策）。"""
    pfc = BigPFC()
    rng = np.random.default_rng(1)
    H, y = [], []
    for s in range(40):
        w = World(seed=s, pits=new_pits(rng), length=60)
        for t in range(200):
            if not w.alive:
                break
            cx = int(round(w.x))
            danger = int(any((cx + c) in w.pits for c in (1, 2, 3)))
            h = pfc.hidden(encode(render(w))[None, :])[0]
            H.append(h); y.append(danger)
            w.step(np.array([1, 0, 0, 1, 0, 0], np.float32))
    H, y = np.array(H), np.array(y)
    if y.min() == y.max():
        print("② 表示：样本里没有危险/安全两类")
        return
    # 简单线性判别（最小二乘）
    A = np.c_[H, np.ones(len(H))]
    wts = np.linalg.lstsq(A, y, rcond=None)[0]
    pred = (A @ wts > 0.5).astype(int)
    acc = float((pred == y).mean())
    base = float(max(y.mean(), 1 - y.mean()))
    print("② 表示：最后一层线性可分的危险准确率 %.3f  基线 %.3f  样本 %d"
          % (acc, base, len(y)))


def test_resolution():
    """不同分辨率下 world 的渲染信息量（坑占多少格）。"""
    for g in (8, 16, 32):
        rng = np.random.default_rng(0)
        w = World(seed=0, pits=new_pits(rng), length=60)
        f = np.zeros((g, g), np.float32)
        col = int(w.x) % g
        print("③ 分辨率 %2d：坑宽 2 格占画面 %.1f%%" % (g, 200.0 / (g * g) * g))


if __name__ == "__main__":
    test_physics()
    test_representation()
    test_resolution()
