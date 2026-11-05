"""泛化测试：一部分 episode 训练，**没见过的 episode** 测试。

判据：测试集上每键准确率 vs "永远按该键"的基线。
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
ROOT = r"F:\一种AGI关联神经元架构\data\smb_all"


def key_vec(a: int) -> np.ndarray:
    v = np.zeros(N_KEY, np.float32)
    for i, (_, bit) in enumerate(KEYS):
        if a & bit:
            v[i] = 1.0
    return v


def episodes(n_train: int, n_test: int, stride: int = 2):
    dirs = sorted(glob.glob(os.path.join(ROOT, "*")))
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(dirs))
    tr = [dirs[i] for i in idx[:n_train]]
    te = [dirs[i] for i in idx[n_train:n_train + n_test]]

    def load(d):
        fs = sorted(glob.glob(os.path.join(d, "*.png")),
                    key=lambda p: int(re.search(r"_f(\d+)_", p).group(1)))
        out = []
        for p in fs[::stride]:
            m = re.search(r"_a(\d+)_", os.path.basename(p))
            if not m:
                continue
            img = Image.open(p).convert("L").resize((8, 8), Image.BILINEAR)
            f = (np.asarray(img, np.float32) / 255.0 > 0.5).astype(np.float32)
            out.append((f, int(m.group(1))))
        return out
    return tr, te, load


def main(n_train=16, n_test=5, stride=2, rule="concave", w_ref=1.0, pw=1.0, lr=0.15):
    tr, te, load = episodes(n_train, n_test, stride)
    pfc = PFC(N_IN)
    Wb = np.zeros((pfc.n4, N_KEY), np.float32)
    u_h = np.zeros(pfc.n4, np.float32)
    u_b = np.zeros(N_KEY, np.float32)
    prev_h = None
    n_tr = 0
    for d in tr:                      # 训练
        for f, a in load(d):
            h = pfc.step(encode(f), learn=True)
            btn = key_vec(a)
            if prev_h is not None:
                upd = np.outer(prev_h - u_h, btn - u_b)
                phi = 1.0 / (1.0 + np.abs(Wb) / w_ref) ** pw if rule == "concave" else 1.0
                Wb += lr * upd * phi - 0.002 * Wb * phi
                np.clip(Wb, -1.0, 1.0, out=Wb)
            u_h += 0.005 * (h - u_h)
            u_b += 0.005 * (btn - u_b)
            prev_h = h
            n_tr += 1
    print("训练帧数: %d（%d 个 episode）" % (n_tr, len(tr)))

    hit = np.zeros(N_KEY, np.float32)
    tot = 0
    freq = np.zeros(N_KEY, np.float32)
    for d in te:                      # 测试（没见过的 episode）
        for f, a in load(d):
            h = pfc.step(encode(f), learn=False)
            btn = key_vec(a)
            drive = (h - u_h) @ Wb
            pred = (drive > 0).astype(np.float32)
            hit += (pred == btn).astype(np.float32)
            freq += btn
            tot += 1
    print("测试帧数: %d（%d 个 episode，都没在训练里出现过）" % (tot, len(te)))
    acc = hit / max(1, tot)
    fq = freq / max(1, tot)
    for i, (nm, _) in enumerate(KEYS):
        print("  %-6s 模型 %.3f  基线 %.3f  出现率 %.2f"
              % (nm, acc[i], max(fq[i], 1 - fq[i]), fq[i]))
    print("总体: 模型 %.3f   基线 %.3f" % (float(acc.mean()),
          float(np.mean([max(x, 1 - x) for x in fq]))))


if __name__ == "__main__":
    main()
