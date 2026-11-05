"""指标：最后一层的编码里，有没有"该不该跳"的信息（线性探针，仅诊断）。

数据：cache_win.npz（帧 + 三个键位的位）；只做前向，不学习。
判据：跳（A）的判别准确率要明显高于基线（类不平衡基线）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.p78_batched import BOTTOM, PER, G, encode_batch, local  # noqa: E402

CACHE = r"F:\一种AGI关联神经元架构\data\cache_win.npz"
N_IN = G * G * PER


def main(n2=768, n3=768, n4=512, chunk=8192):
    d = np.load(CACHE)
    F, A = d["frames"], d["actions"]
    N = len(F)
    F2, F3, F4 = local(n2, N_IN, 8), local(n3, n2, 8), local(n4, n3, 8)
    H_all = []
    t2 = t3 = None
    for s in range(0, N, chunk):
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
        idx = np.argpartition(-d4, 128, axis=1)[:, :128]
        np.put_along_axis(H, idx, 1.0, axis=1)
        H_all.append(H)
    H = np.concatenate(H_all)
    print("编码数 %d，维度 %d" % (H.shape[0], H.shape[1]))

    # 三个键位分别做线性探针
    tests = (("right", 1), ("B", 2), ("A/跳", 4))
    n = len(H)
    tr = slice(0, n // 2)
    te = slice(n // 2, n)
    A0 = np.c_[H, np.ones(n)]
    for name, bit in tests:
        y = ((A & bit) > 0).astype(np.float32)
        w = np.linalg.lstsq(A0[tr], y[tr], rcond=None)[0]
        pred = (A0[te] @ w > 0.5).astype(np.float32)
        acc = float((pred == y[te]).mean())
        base = float(max(y[te].mean(), 1 - y[te].mean()))
        print("  %-6s 探针准确 %.3f   基线 %.3f   该键占比 %.2f"
              % (name, acc, base, float(y.mean())))


if __name__ == "__main__":
    main()
