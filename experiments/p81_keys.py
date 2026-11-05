"""直接按键：最后一层 → 三个键（右/B/跳）的赫布读出（凹形增长 + 统一降权 + 弱删除）。

判据：每键平衡准确率 > 0.65。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.p78_batched import G, PER, encode_batch, local  # noqa: E402

CACHE = r"F:\一种AGI关联神经元架构\data\cache_win.npz"
N_IN = G * G * PER
KEYS = (("right", 1), ("B", 2), ("A/跳", 4))


def bal_acc(score, y):
    best = 0.0
    for th in np.linspace(np.percentile(score, 2), np.percentile(score, 98), 60):
        p = score > th
        pos = max(1, (y > 0.5).sum())
        neg = max(1, (y < 0.5).sum())
        b = 0.5 * ((p & (y > 0.5)).sum() / pos + ((~p) & (y < 0.5)).sum() / neg)
        best = max(best, float(b))
    return best


def main(epochs=30, lr=0.05, chunk=4096, n2=768, n3=768, n4=512,
         decay=0.02, concave=True, scale=1.0, wpos=1.0):
    d = np.load(CACHE)
    F, A = d["frames"], d["actions"]
    N = len(F)
    print("帧数 %d" % N, flush=True)
    F2, F3, F4 = local(n2, N_IN, 8), local(n3, n2, 8), local(n4, n3, 8)
    t2 = t3 = None
    W = np.zeros((n4, len(KEYS)), np.float32)
    u_h = np.zeros(n4, np.float32)
    u_y = np.zeros(len(KEYS), np.float32)
    t0 = time.time()
    for ep in range(epochs):
        for s in range(0, N - chunk, chunk):
            X = encode_batch(F[s:s + chunk])
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
            k = np.argpartition(-d4, 128, axis=1)[:, :128]
            np.put_along_axis(H, k, 1.0, axis=1)
            Y = np.stack([((A[s:s + chunk] & bit) > 0).astype(np.float32)
                          for _, bit in KEYS], 1)
            # 只改增长/下降机制：delta（按预测误差增长）+ Oja 式衰减（按输出强度）
            Hc = H[:-1] - u_h
            Yc = Y[1:] - u_y
            if wpos != 1.0:                    # 稀有键加权（属于增长机制）
                Yc = Yc * np.where(Y[1:] > 0.5, wpos, 1.0).astype(np.float32)
            pred = Hc @ W                      # 当前预测（突触局部可得）
            err = Yc - pred                    # 预测误差
            dW = (Hc.T @ err) / chunk          # 按误差增长
            oja = (pred ** 2).mean(0)          # 输出强度（越大越衰减）
            W += lr * dW - decay * (W * oja[None, :])
            # 不用 ±1 硬裁剪：按列归一化，方向自由、幅度受控
            nrm = np.sqrt((W ** 2).sum(0, keepdims=True)) + 1e-6
            W *= (scale / nrm)
            W[np.abs(W) < 0.05 * np.abs(W).max(0, keepdims=True)] = 0.0
            u_h += 0.005 * (H.mean(0) - u_h)
            u_y += 0.005 * (Y.mean(0) - u_y)
        print("第%d轮 %.0fs" % (ep + 1, time.time() - t0), flush=True)

    # 评测：后半段
    res = {}
    n = 0
    scores = [[] for _ in KEYS]
    ys = [[] for _ in KEYS]
    for s in range(N // 2, N - chunk, chunk):
        X = encode_batch(F[s:s + chunk])
        L2 = (X @ F2.T - t2 > 0).astype(np.float32)
        L3 = (L2 @ F3.T - t3 > 0).astype(np.float32)
        d4 = L3 @ F4.T
        H = np.zeros_like(d4)
        k = np.argpartition(-d4, 128, axis=1)[:, :128]
        np.put_along_axis(H, k, 1.0, axis=1)
        sc = (H - u_h) @ W
        Y = np.stack([((A[s:s + chunk] & bit) > 0).astype(np.float32)
                      for _, bit in KEYS], 1)
        for i in range(len(KEYS)):
            scores[i].append(sc[:, i]); ys[i].append(Y[:, i])
        n += len(H)
    for i, (nm, _) in enumerate(KEYS):
        sc = np.concatenate(scores[i]); y = np.concatenate(ys[i])
        res[nm] = (bal_acc(sc, y), float(y.mean()))
        print("  %-6s 平衡准确率 %.3f  占比 %.2f" % (nm, res[nm][0], res[nm][1]))


if __name__ == "__main__":
    main()
