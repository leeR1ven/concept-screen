"""量最后一层对不同画面的区分度：同类重叠 c、异类重叠 o（照 THEORY §1 定义）。"""
import sys

import numpy as np

sys.path.insert(0, ".")
from experiments.p51_screen2key import N_BTN, PFC, N_IN, RNG, encode, synth_frame


def overlap(a, b):
    na, nb = float(a.sum()), float(b.sum())
    if na == 0 or nb == 0:
        return 0.0
    return float((a * b).sum() / np.sqrt(na * nb))


def main(n: int = 300):
    pfc = PFC(N_IN)
    hs, cls = [], []
    for _ in range(n):
        f, btn = synth_frame(RNG)
        h = pfc.step(encode(f), learn=False)
        hs.append(h.copy())
        cls.append(int(np.argmax(btn)))
    hs, cls = np.array(hs), np.array(cls)
    # 去掉"什么都亮"的公共部分（THEORY §2.2：去掉后仍重叠才是真的像）
    base = hs.mean(0, keepdims=True)
    hs_c = np.clip(hs - base, 0, None)
    print("最后一层: n4=%d  m=%d  每拍亮 %.2f  去公共后平均亮 %.1f" %
          (pfc.n4, pfc.m, hs.sum(1).mean(), hs_c.sum(1).mean()))

    same, diff = [], []
    for i in range(n):
        for j in range(i + 1, min(n, i + 40)):
            v = overlap(hs_c[i], hs_c[j])
            (same if cls[i] == cls[j] else diff).append(v)
    print("同类重叠 c = %.3f   异类重叠 o = %.3f   比值 c/o = %.2f"
          % (np.mean(same), np.mean(diff), np.mean(same) / max(1e-6, np.mean(diff))))
    print("=> 能学的前提是 c 明显大于 o（比值越大越好，比值≈1 就是没有区分度）")


if __name__ == "__main__":
    main()
