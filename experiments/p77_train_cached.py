"""在缓存上训练（学"下一帧底部一行"），并按帧数报告命中率曲线。

数据：data/cache_win.npz（只含通关局，已丢末尾 30 帧）
判据：命中率随帧数上升 = 学习有效。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.p51_screen2key import N_IN, PFC, encode  # noqa: E402

CACHE = r"F:\一种AGI关联神经元架构\data\cache_win.npz"
G = 8
BOTTOM = slice((G - 1) * G * 8, G * G * 8)


def to_frame(flat: np.ndarray) -> np.ndarray:
    return flat.reshape(G, G).astype(np.float32)


def main(epochs=3, lr=0.05, report=5000):
    d = np.load(CACHE)
    F, A = d["frames"], d["actions"]
    n = len(F)
    print("帧数 %d" % n, flush=True)
    pfc = PFC(N_IN)
    Wnext = np.zeros((pfc.n4, N_IN), np.float32)
    u_h = np.zeros(pfc.n4, np.float32)
    u_x = np.zeros(N_IN, np.float32)
    hit = tot = step = 0
    t0 = time.time()
    for ep in range(epochs):
        prev_h = None
        for i in range(n):
            x = encode(to_frame(F[i]))
            h = pfc.step(x, learn=True)
            if prev_h is not None:
                a = prev_h - u_h
                tgt = x - u_x
                phi = 1.0 / (1.0 + np.abs(Wnext))
                Wnext += lr * np.outer(a, tgt) * phi
                np.clip(Wnext, -1.0, 1.0, out=Wnext)
                pred = prev_h @ Wnext
                truth, got = x[BOTTOM], pred[BOTTOM]
                if truth.sum() > 0:
                    hit += int(np.argmax(np.abs(got)) == np.argmax(np.abs(truth)))
                    tot += 1
            u_h += 0.005 * (h - u_h)
            u_x += 0.005 * (x - u_x)
            prev_h = h
            step += 1
            if step % report == 0:
                print("第%d轮 帧 %7d  底部行命中 %.3f  用时 %.0fs"
                      % (ep + 1, step, hit / max(1, tot), time.time() - t0), flush=True)
        print("—— 第 %d 轮结束：帧 %d，命中 %.3f"
              % (ep + 1, step, hit / max(1, tot)), flush=True)
    np.save(CACHE.replace("cache_win.npz", "Wnext_win.npy"), Wnext)


if __name__ == "__main__":
    main()
