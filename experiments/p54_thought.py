"""想法回路（正确版）：最后一层 → 下一拍第一层副本 的赫布连接 Wnext 学 A→B。

检验：画面消失后，模型能否靠 Wnext 点亮"下一拍该有的副本"，继续预测按键。
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from experiments.p51_screen2key import N_BTN, N_IN, PFC, RNG, encode, synth_frame


def run(ticks: int = 12000, use_next: bool = True, blank_after: float = 0.6,
        lr_next: float = 0.20):
    pfc = PFC(N_IN)
    Wb = np.zeros((pfc.n4, N_BTN), np.float32)      # 最后一层 → 按键
    Wnext = np.zeros((pfc.n4, N_IN), np.float32)    # 最后一层 → 下一拍第一层
    u_h = np.zeros(pfc.n4, np.float32)
    u_x = np.zeros(N_IN, np.float32)
    u_b = np.zeros(N_BTN, np.float32)
    y = np.zeros(N_IN, np.float32)
    u_d = np.zeros(N_IN, np.float32)

    hit = tot = 0
    cnt = np.zeros(N_BTN)
    prev_h = None
    y_sums = []
    blank_from = int(ticks * blank_after)
    for t in range(ticks):
        f, btn = synth_frame(RNG)
        x = encode(f)
        if t >= blank_from:
            x = np.zeros_like(x)
        l1 = np.maximum(x, y)
        h = pfc.step(l1, learn=True)

        learning = t < blank_from          # 空屏期间只预测、不学习
        if prev_h is not None:
            drive = (prev_h - u_h) @ Wb
            pred = int(np.argmax(drive))
            if t >= blank_from:
                tot += 1
                if pred == int(np.argmax(btn)):
                    hit += 1
                cnt[int(np.argmax(btn))] += 1
            b = (btn > 0.5).astype(np.float32) - u_b
            if learning: Wb += 0.15 * np.outer(prev_h - u_h, b) - 0.002 * Wb
            np.clip(Wb, -1.0, 1.0, out=Wb)
            if use_next and learning:
                a = prev_h - u_h
                tgt = x - u_x
                Wnext += lr_next * np.outer(a, tgt) - 0.002 * Wnext
                np.clip(Wnext, -1.0, 1.0, out=Wnext)

        if use_next:
            d = h @ (pfc.G + Wnext)
            dev = d - u_d
            thr = np.quantile(dev, 1.0 - min(0.25, pfc.m / N_IN))
            y = (dev > max(thr, 0)).astype(np.float32)
            u_d += 0.005 * (d - u_d)
            y_sums.append(float(y.sum()))

        u_h += 0.005 * (h - u_h)
        u_x += 0.005 * (x - u_x)
        u_b += 0.005 * (btn - u_b)
        prev_h = h

    acc = hit / max(1, tot)
    y_mean = float(np.mean(y_sums)) if y_sums else 0.0
    print("空屏期间想法平均点亮 %.1f / %d\n画面消失后准确率: %.3f   基线: %.3f   样本 %d   Wnext=%s"
          % (y_mean, N_IN, acc, float(cnt.max() / max(1, tot)), tot, use_next))


if __name__ == "__main__":
    run(use_next=False)
    run(use_next=True)
