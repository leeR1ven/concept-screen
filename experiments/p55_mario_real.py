"""真实数据：马里奥画面 → 按键（用 smbdataset 的 PNG + 文件名里的动作码）。

动作码 a<0..255> 的位（MSB→LSB）：A, up, left, B, start, right, down, select
只取 6 个常用键：right(4) left(32) A/跳(128) B/跑(16) down(2) up(64)
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.p51_screen2key import N_IN, PFC, encode  # noqa: E402

KEYS = (("right", 4), ("left", 32), ("A", 128), ("B", 16), ("down", 2), ("up", 64))
N_KEY = len(KEYS)
GRID = 8


def parse(fn: str) -> int | None:
    m = re.search(r"_a(\d+)_", os.path.basename(fn))
    return int(m.group(1)) if m else None


def load_frames(pattern: str, limit: int = 3000):
    files = sorted(glob.glob(pattern), key=lambda p: int(re.search(r"_f(\d+)_", p).group(1)))
    out = []
    for p in files[:limit]:
        a = parse(p)
        if a is None:
            continue
        img = Image.open(p).convert("L").resize((GRID, GRID), Image.BILINEAR)
        f = (np.asarray(img, np.float32) / 255.0 > 0.5).astype(np.float32)
        out.append((f, a))
    return out


def keys_vec(a: int) -> np.ndarray:
    v = np.zeros(N_KEY, np.float32)
    for i, (_, bit) in enumerate(KEYS):
        if a & bit:
            v[i] = 1.0
    return v


def main(pattern: str, ticks: int | None = None, lr: float = 0.15,
         rule: str = "concave", w_ref: float = 1.0, pw: float = 1.0):
    data = load_frames(pattern)
    if not data:
        print("没有找到帧:", pattern)
        return
    print("帧数:", len(data))
    acts = np.array([a for _, a in data])
    print("各键出现率:", {n: round(float(((acts & b) > 0).mean()), 2) for n, b in KEYS})

    pfc = PFC(N_IN)
    Wb = np.zeros((pfc.n4, N_KEY), np.float32)
    u_h = np.zeros(pfc.n4, np.float32)
    u_b = np.zeros(N_KEY, np.float32)
    prev_h = None
    hit = tot = 0
    cnt = np.zeros(N_KEY)
    per_hit = np.zeros(N_KEY, np.float32)
    per_tot = 0
    n = ticks or len(data)
    for t in range(min(n, len(data))):
        f, a = data[t]
        h = pfc.step(encode(f), learn=True)
        btn = keys_vec(a)
        if prev_h is not None and t > n // 2:
            drive = (prev_h - u_h) @ Wb          # 每个键一个打分
            pred = (drive > 0).astype(np.float32)  # 独立判断"按/不按"
            tot += 1
            per_hit += (pred == btn).astype(np.float32)
            per_tot += 1
        if prev_h is not None:
            b = btn - u_b
            upd = np.outer(prev_h - u_h, b)
            if rule == "concave":
                # 凹形增长：越高的权重越难涨（架构规则），减弱/降权同样变慢
                phi = 1.0 / (1.0 + np.abs(Wb) / w_ref) ** pw
                Wb += lr * upd * phi - 0.002 * Wb * phi
            else:
                Wb += lr * upd - 0.002 * Wb
            np.clip(Wb, -1.0, 1.0, out=Wb)
        u_h += 0.005 * (h - u_h)
        u_b += 0.005 * (btn - u_b)
        prev_h = h

    freq = np.array([float(((acts & b) > 0).mean()) for _, b in KEYS])
    acc = per_hit / max(1, per_tot)
    print("逐键准确率（模型 vs 永远按该键的基线）:")
    for i, (nm, _) in enumerate(KEYS):
        base = max(freq[i], 1 - freq[i])
        print("  %-6s 模型 %.3f   基线 %.3f   出现率 %.2f" % (nm, acc[i], base, freq[i]))
    print("总体: 模型 %.3f   基线 %.3f   样本 %d"
          % (float(acc.mean()), float(np.mean([max(f, 1 - f) for f in freq])), per_tot))


if __name__ == "__main__":
    main(r"F:\一种AGI关联神经元架构\data\smb\**\*.png")
