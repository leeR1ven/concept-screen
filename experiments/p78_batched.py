"""批处理版：一次算一批帧的最后一层，协方差赫布用矩阵乘一次性更新。

数学上与逐帧等价（只是把 Σ 外积写成矩阵乘），速度提升几十倍。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.p51_screen2key import N_IN  # noqa: E402

CACHE = r"F:\一种AGI关联神经元架构\data\cache_win.npz"
G = 8
LV = 4
PER = LV * 2
BOTTOM = slice((G - 1) * G * 8, G * G * 8)
RNG = np.random.default_rng(0)


def encode_batch(F: np.ndarray) -> np.ndarray:
    n = F.shape[0]
    X = np.zeros((n, N_IN), np.float32)
    flat = (np.arange(G * G) * PER)
    lv = np.clip((F.reshape(n, -1).astype(np.int32)) * LV, 0, LV - 1)
    rows = np.arange(n)[:, None]
    X[rows, flat[None, :] + lv] = 1.0
    X[rows, flat[None, :] + LV + (LV - 1 - lv)] = 1.0
    return X


def local(n_out, n_in, k):
    po = np.arange(n_out)[:, None] / max(1, n_out - 1)
    pi = np.arange(n_in)[None, :] / max(1, n_in - 1)
    idx = np.argsort(np.abs(po - pi), axis=1)[:, :k]
    M = np.zeros((n_out, n_in), np.float32)
    for o in range(n_out):
        M[o, idx[o]] = RNG.uniform(0.5, 1.5, k)
    return M


def main(chunk=4096, epochs=6, lr=0.05, n2=768, n3=768, n4=512):
    d = np.load(CACHE)
    F = d["frames"]
    N = len(F)
    print("帧数 %d" % N, flush=True)
    F2, F3, F4 = local(n2, N_IN, 8), local(n3, n2, 8), local(n4, n3, 8)
    t2 = np.zeros(n2, np.float32)
    t3 = np.zeros(n3, np.float32)
    Wnext = np.zeros((n4, N_IN), np.float32)
    u_h = np.zeros(n4, np.float32)
    u_x = np.zeros(N_IN, np.float32)
    m = 128
    hit = tot = 0
    t0 = time.time()
    for ep in range(epochs):
        for s in range(0, N - chunk, chunk):
            X = encode_batch(F[s:s + chunk])
            d2 = X @ F2.T
            if ep == 0 and s == 0:
                t2[:] = np.quantile(d2, 0.7)
            L2 = (d2 - t2 > 0).astype(np.float32)
            d3 = L2 @ F3.T
            if ep == 0 and s == 0:
                t3[:] = np.quantile(d3, 0.7)
            L3 = (d3 - t3 > 0).astype(np.float32)
            d4 = L3 @ F4.T
            H = np.zeros_like(d4)
            idx = np.argpartition(-d4, m, axis=1)[:, :m]
            np.put_along_axis(H, idx, 1.0, axis=1)
            # 协方差赫布（整批一次）
            dW = (H[:-1] - u_h).T @ (X[1:] - u_x) / chunk
            phi = 1.0 / (1.0 + np.abs(Wnext))
            Wnext += lr * dW * phi
            np.clip(Wnext, -1.0, 1.0, out=Wnext)
            # 架构要求的统一降权：越弱越容易被降、被删（凹形）
            Wnext *= (1.0 - 0.02 * phi)
            Wnext[np.abs(Wnext) < 0.005] = 0.0
            pred = H[:-1] @ Wnext
            truth = X[1:][:, BOTTOM]
            got = pred[:, BOTTOM]
            ok = truth.sum(1) > 0
            hit += int((np.argmax(np.abs(got[ok]), 1) ==
                        np.argmax(np.abs(truth[ok]), 1)).sum())
            tot += int(ok.sum())
            u_h += 0.005 * (H.mean(0) - u_h)
            u_x += 0.005 * (X.mean(0) - u_x)
        print("第%d轮结束：累计帧 %d，底部行命中 %.3f，用时 %.0fs"
              % (ep + 1, (s + chunk), hit / max(1, tot), time.time() - t0), flush=True)
    print("最终命中 %.3f（%d 样本）" % (hit / max(1, tot), tot))


if __name__ == "__main__":
    main()
