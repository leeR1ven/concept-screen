"""大配置长训：16×16 / 中间层 1024 / 最后一层 2048 / top-m 512。

只学"看到什么 → 下一拍按什么键"，一条赫布规则（协方差 + 列归一化 + 弱衰减）。
进度每轮打印，便于看长期趋势。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.p82_resolution import KEYS, bal_acc, build_cache, encode_batch, local  # noqa: E402

CACHE = r"F:\一种AGI关联神经元架构\data\cache_win16.npz"
G = 16
LV = 4
N_IN = G * G * LV * 2
M = 512
N_MID = 1024
N4 = 2048


def main(epochs=40, lr=0.05, decay=0.005, chunk=2048):
    build_cache(G)
    d = np.load(CACHE)
    F, A = d["frames"], d["actions"]
    N = len(F)
    print("帧数 %d  输入 %d  中间 %d  最后一层 %d  top-m %d"
          % (N, N_IN, N_MID, N4, M), flush=True)
    F2, F3 = local(N_MID, N_IN, 8), local(N_MID, N_MID, 8)
    F4 = local(N4, N_MID, 8)
    t2 = t3 = None
    W = np.zeros((N4, len(KEYS)), np.float32)
    u_h = np.zeros(N4, np.float32)
    u_y = np.zeros(len(KEYS), np.float32)
    t0 = time.time()
    for ep in range(epochs):
        for s in range(0, N - chunk, chunk):
            X = encode_batch(F[s:s + chunk], G)
            d2 = X @ F2.T
            if t2 is None:
                t2 = np.quantile(d2, 0.7)
            L2 = (d2 - t2 > 0).astype(np.float32)
            d3 = L2 @ F3.T
            if t3 is None:
                t3 = np.quantile(d3, 0.7)
            L3 = (d3 - t3 > 0).astype(np.float32)
            d4 = L3 @ F4.T
            H = np.zeros_like(d4)
            k = np.argpartition(-d4, M, axis=1)[:, :M]
            np.put_along_axis(H, k, 1.0, axis=1)
            Y = np.stack([((A[s:s + chunk] & bit) > 0).astype(np.float32)
                          for _, bit in KEYS], 1)
            dW = (H[:-1] - u_h).T @ (Y[1:] - u_y) / chunk
            W += lr * dW - decay * W
            nrm = np.sqrt((W ** 2).sum(0, keepdims=True)) + 1e-6
            W *= (1.0 / nrm)
            W[np.abs(W) < 0.05 * np.abs(W).max(0, keepdims=True)] = 0.0
            u_h += 0.005 * (H.mean(0) - u_h)
            u_y += 0.005 * (Y.mean(0) - u_y)
        print("第%d轮  %.0fs" % (ep + 1, time.time() - t0), flush=True)

    sc_all = [[] for _ in KEYS]
    y_all = [[] for _ in KEYS]
    for s in range(N // 2, N - chunk, chunk):
        X = encode_batch(F[s:s + chunk], G)
        L2 = (X @ F2.T - t2 > 0).astype(np.float32)
        L3 = (L2 @ F3.T - t3 > 0).astype(np.float32)
        d4 = L3 @ F4.T
        H = np.zeros_like(d4)
        k = np.argpartition(-d4, M, axis=1)[:, :M]
        np.put_along_axis(H, k, 1.0, axis=1)
        sc = (H - u_h) @ W
        Y = np.stack([((A[s:s + chunk] & bit) > 0).astype(np.float32)
                      for _, bit in KEYS], 1)
        for i in range(len(KEYS)):
            sc_all[i].append(sc[:, i]); y_all[i].append(Y[:, i])
    for i, (nm, _) in enumerate(KEYS):
        sc = np.concatenate(sc_all[i]); y = np.concatenate(y_all[i])
        print("  %-6s 平衡准确率 %.3f" % (nm, bal_acc(sc, y)), flush=True)


if __name__ == "__main__":
    main()
