"""按架构规则校准力细胞门槛：每个神经元的门槛由它自己收到的信号分布决定。

架构（docs/ARCHITECTURE.md 二.4）："每个神经元的门槛按它自己收到的信号来定。"
我们的力细胞用的是固定阶梯 K×(0.1~0.9)，实测输入总和远高于它 → 10 档全亮。
这里改成：先空跑 N 拍，记录每个力细胞的实际驱动，把门槛设成该分布的百分位。
"""
from __future__ import annotations

import numpy as np


def calibrate_force(eng, brain, sp, ticks: int = 40, levels: int | None = None) -> dict:
    """空跑若干拍，按每个力细胞的驱动分布设置它的门槛（温度计重新标定）。"""
    from agi.dog_tasks import _obs

    n_force = int(sp.n_force)
    force_cells = [np.asarray(brain.motor["force"][m], dtype=np.int64) for m in range(sp.n_muscle)]
    all_cells = np.concatenate(force_cells)

    # 每拍记录力细胞的驱动（= 上游激活 × 权重 的和）
    drive = np.zeros((ticks, all_cells.size), np.float32)
    for t in range(ticks):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, None)))
        eng.step()
        act = np.asarray(eng.act)
        src, dst, wp = np.asarray(eng.src), np.asarray(eng.dst), np.asarray(eng.wp)
        for k, c in enumerate(all_cells):
            m = dst == c
            drive[t, k] = float((act[src[m]] * wp[m]).sum())

    # 每块肌肉内部 10 档：按驱动分布的分位数设门槛（从低到高）
    qs = np.linspace(20, 95, n_force)          # 20%, 28%, ..., 95%
    thr = np.zeros(all_cells.size, np.float32)
    for m in range(sp.n_muscle):
        sl = slice(m * n_force, (m + 1) * n_force)
        for k in range(n_force):
            d = drive[:, m * n_force + k]       # 这个细胞自己的驱动序列
            q = 20 + 70 * k / max(1, n_force - 1)
            thr[m * n_force + k] = float(np.percentile(d, q) + 1e-3)
    eng.set_thr(all_cells, thr=thr, soft=np.full(all_cells.size, 0.35, np.float32))
    return {"drive_mean": float(drive.mean()), "thr_min": float(thr.min()),
            "thr_max": float(thr.max())}
