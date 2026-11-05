"""真实马里奥 + 动作压缩：256 种按键组合 → 4 个宏动作。

宏动作（按优先级）：跳(A) / 跑(B+右) / 右 / 停
判据：逐宏准确率 vs "永远猜该宏"的基线。
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.p51_screen2key import N_IN, PFC, encode  # noqa: E402

MACROS = ("jump", "run", "right", "idle")
ROOT = r"F:\一种AGI关联神经元架构\data\smb_all"


def macro(a: int) -> int:
    if a & 128:
        return 0
    if (a & 16) and (a & 4):
        return 1
    if a & 4:
        return 2
    return 3


def load(episodes, stride=3):
    dirs = sorted(glob.glob(os.path.join(ROOT, "*")))[:episodes]
    for d in dirs:
        fs = sorted(glob.glob(os.path.join(d, "*.png")),
                    key=lambda p: int(re.search(r"_f(\d+)_", p).group(1)))
        for p in fs[::stride]:
            m = re.search(r"_a(\d+)_", os.path.basename(p))
            if not m:
                continue
            img = Image.open(p).convert("L").resize((8, 8), Image.BILINEAR)
            f = (np.asarray(img, np.float32) / 255.0 > 0.5).astype(np.float32)
            yield f, macro(int(m.group(1)))


def main(episodes=12, lr=0.35, w_ref=1.0, pw=1.0):
    pfc = PFC(N_IN)
    Wb = np.zeros((pfc.n4, len(MACROS)), np.float32)
    u_h = np.zeros(pfc.n4, np.float32)
    u_b = np.zeros(len(MACROS), np.float32)
    prev_h = None
    n = 0
    for f, m in load(episodes):
        h = pfc.step(encode(f), learn=True)
        tgt = np.zeros(len(MACROS), np.float32); tgt[m] = 1.0
        if prev_h is not None:
            phi = 1.0 / (1.0 + np.abs(Wb) / w_ref) ** pw
            Wb += lr * np.outer(prev_h - u_h, tgt - u_b) * phi - 0.002 * Wb * phi
            np.clip(Wb, -1.0, 1.0, out=Wb)
        u_h += 0.005 * (h - u_h)
        u_b += 0.005 * (tgt - u_b)
        prev_h = h
        n += 1
    print("训练帧数:", n)

    hit = np.zeros(len(MACROS)); cnt = np.zeros(len(MACROS)); tot = 0
    for f, m in load(episodes, stride=7):
        h = pfc.step(encode(f), learn=False)
        pred = int(np.argmax((h - u_h) @ Wb))
        hit[m] += 1.0 if pred == m else 0.0      # 按"真值类别"统计召回
        cnt[m] += 1
        tot += 1
    print("测试帧数:", tot)
    for i, nm in enumerate(MACROS):
        base = float(cnt[i] / max(1, tot))
        print("  %-6s 召回 %.3f   该宏占比 %.2f   （随机/多数基线 %.2f）"
              % (nm, float(hit[i] / max(1, cnt[i])), base, max(base, 1 - base)))


if __name__ == "__main__":
    main()
