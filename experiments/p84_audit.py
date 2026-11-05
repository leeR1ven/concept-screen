"""审计版评测：①按 episode 分开（前 70% 训练、后 30% 测试，互不重叠）；
②m 现算（= 第一层实际亮数）；③阈值在全量样本上标定。

目的：拿到没有泄漏的数字，再谈模型好坏。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.p82_resolution import KEYS, bal_acc, build_cache, encode_batch, local  # noqa: E402

CACHE8 = r"F:\一种AGI关联神经元架构\data\cache_win.npz"


def run(G, cache, epochs=30, lr=0.05, decay=0.005, n_mid=768, n4=512,
        chunk=2048, train_frac=0.7, seed=0):
    rng = np.random.default_rng(seed)
    d = np.load(cache)
    F, A = d["frames"], d["actions"]
    N = len(F)
    n_tr = int(N * train_frac)
    print("G=%d 总帧 %d：训练 %d / 测试 %d（不同 episode 段）"
          % (G, N, n_tr, N - n_tr), flush=True)
    n_in = G * G * 8
    F2, F3, F4 = local(n_mid, n_in, 8), local(n_mid, n_mid, 8), local(n4, n_mid, 8)

    # 阈值：在 2 万帧的随机样本上标定（不使用测试段统计）
    samp = rng.choice(n_tr, size=min(20000, n_tr), replace=False)
    samp.sort()
    Xs = encode_batch(F[samp], G)
    d2s = Xs @ F2.T
    t2 = np.quantile(d2s, 0.7)
    L2s = (d2s - t2 > 0).astype(np.float32)
    d3s = L2s @ F3.T
    t3 = np.quantile(d3s, 0.7)
    L3s = (d3s - t3 > 0).astype(np.float32)
    d4s = L3s @ F4.T
    m = int(np.round(Xs.sum(1).mean()))          # m 现算 = 第一层亮数
    print("阈值 t2=%.3f t3=%.3f  m=%d（现算）" % (t2, t3, m), flush=True)

    W = np.zeros((n4, len(KEYS)), np.float32)
    u_h = np.zeros(n4, np.float32)
    u_y = np.zeros(len(KEYS), np.float32)
    t0 = time.time()
    for ep in range(epochs):
        for s in range(0, n_tr - chunk, chunk):
            X = encode_batch(F[s:s + chunk], G)
            d2 = X @ F2.T
            L2 = (d2 - t2 > 0).astype(np.float32)
            d3 = L2 @ F3.T
            L3 = (d3 - t3 > 0).astype(np.float32)
            d4 = L3 @ F4.T
            H = np.zeros_like(d4)
            k = np.argpartition(-d4, m, axis=1)[:, :m]
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
        if (ep + 1) % 10 == 0:
            print("  第%d轮 %.0fs" % (ep + 1, time.time() - t0), flush=True)

    sc_all = [[] for _ in KEYS]
    y_all = [[] for _ in KEYS]
    for s in range(n_tr, N - chunk, chunk):
        X = encode_batch(F[s:s + chunk], G)
        L2 = (X @ F2.T - t2 > 0).astype(np.float32)
        L3 = (L2 @ F3.T - t3 > 0).astype(np.float32)
        d4 = L3 @ F4.T
        H = np.zeros_like(d4)
        k = np.argpartition(-d4, m, axis=1)[:, :m]
        np.put_along_axis(H, k, 1.0, axis=1)
        sc = (H - u_h) @ W
        Y = np.stack([((A[s:s + chunk] & bit) > 0).astype(np.float32)
                      for _, bit in KEYS], 1)
        for i in range(len(KEYS)):
            sc_all[i].append(sc[:, i]); y_all[i].append(Y[:, i])
    out = []
    for i, (nm, _) in enumerate(KEYS):
        out.append((nm, bal_acc(np.concatenate(sc_all[i]), np.concatenate(y_all[i]))))
    return out


if __name__ == "__main__":
    print("8×8 审计：", run(8, CACHE8, epochs=30))
