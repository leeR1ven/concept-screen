"""分辨率对照实验：同一批帧，8×8 vs 16×16，同样的赫布读出与参数。

只改分辨率，其它完全不变 —— 用它回答"是不是看不清图案"。
"""
from __future__ import annotations

import glob
import os
import re
import sys
import time

import numpy as np
from PIL import Image

ROOT = r"F:\一种AGI关联神经元架构\data\smb_all"
CACHE_T = r"F:\一种AGI关联神经元架构\data\cache_win{g}.npz"
KEYS = (("right", 1), ("B", 2), ("A", 4))
RNG = np.random.default_rng(0)


def build_cache(G=16, stride=2):
    CACHE = CACHE_T.format(g=G)
    if os.path.exists(CACHE):
        return
    dirs = [d for d in sorted(glob.glob(os.path.join(ROOT, "*")))
            if "win" in os.path.basename(d).lower()]
    frames, acts = [], []
    t0 = time.time()
    for d in dirs:
        fs = sorted(glob.glob(os.path.join(d, "*.png")),
                    key=lambda p: int(re.search(r"_f(\d+)_", p).group(1)))[:-30][::stride]
        for p in fs:
            m = re.search(r"_a(\d+)_", os.path.basename(p))
            if not m:
                continue
            img = Image.open(p).convert("L").resize((G, G), Image.BILINEAR)
            frames.append((np.asarray(img, np.uint8) > 127).astype(np.uint8).ravel())
            a = int(m.group(1))
            acts.append(sum(1 << k for k, (_, bit) in enumerate(KEYS) if a & bit))
    np.savez_compressed(CACHE, frames=np.stack(frames), actions=np.array(acts, np.uint8))
    print("缓存完成 %d 帧 %.0fs" % (len(frames), time.time() - t0), flush=True)


def encode_batch(F, G, LV=4):
    n = F.shape[0]
    per = LV * 2
    X = np.zeros((n, G * G * per), np.float32)
    flat = np.arange(G * G) * per
    lv = np.clip(F.reshape(n, -1).astype(np.int32) * LV, 0, LV - 1)
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


def bal_acc(score, y):
    best = 0.0
    for th in np.linspace(np.percentile(score, 2), np.percentile(score, 98), 60):
        p = score > th
        npos = int((y > .5).sum()); nneg = int((y < .5).sum())
        b = 0.5 * (float((p & (y > .5)).sum()) / max(1, npos) +
                   float(((~p) & (y < .5)).sum()) / max(1, nneg))
        best = max(best, float(b))
    return best


def run(G, path, epochs=15, lr=0.05, decay=0.005, n_mid=768, n4=512, m=128,
        limit=None):
    d = np.load(path)
    F, A = d["frames"], d["actions"]
    if limit:
        F, A = F[:limit], A[:limit]
    N = len(F)
    n_in = G * G * 8
    F2, F3, F4 = local(n_mid, n_in, 8), local(n_mid, n_mid, 8), local(n4, n_mid, 8)
    t2 = t3 = None
    W = np.zeros((n4, 3), np.float32)
    u_h = np.zeros(n4, np.float32)
    u_y = np.zeros(3, np.float32)
    chunk = 4096
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

    acc = []
    n = 0
    sc_all = [[] for _ in KEYS]
    y_all = [[] for _ in KEYS]
    for s in range(N // 2, N - chunk, chunk):
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
        for i in range(3):
            sc_all[i].append(sc[:, i]); y_all[i].append(Y[:, i])
    for i, (nm, _) in enumerate(KEYS):
        acc.append((nm, bal_acc(np.concatenate(sc_all[i]), np.concatenate(y_all[i]))))
    return acc


if __name__ == "__main__":
    for G in (16, 24):
        build_cache(G)
        print("%d×%d 结果：" % (G, G), run(G, CACHE_T.format(g=G), epochs=10), flush=True)
