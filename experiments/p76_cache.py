"""把通关 episode 的画面一次性降采样缓存成 npy，之后训练就很快。

只取文件名含 win 的局；每局末尾 30 帧丢弃。
输出：data/cache_win.npz（frames uint8 [N,64], actions uint8 [N]）
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
OUT = r"F:\一种AGI关联神经元架构\data\cache_win.npz"
G = 8
KEYS = (("right", 4), ("B", 16), ("A", 128))


def main(stride=2, every=5000):
    dirs = [d for d in sorted(glob.glob(os.path.join(ROOT, "*")))
            if "win" in os.path.basename(d).lower()]
    print("通关局数:", len(dirs), flush=True)
    frames, acts = [], []
    t0 = time.time()
    for i, d in enumerate(dirs):
        fs = sorted(glob.glob(os.path.join(d, "*.png")),
                    key=lambda p: int(re.search(r"_f(\d+)_", p).group(1)))
        fs = fs[:-30][::stride]
        for p in fs:
            m = re.search(r"_a(\d+)_", os.path.basename(p))
            if not m:
                continue
            a = int(m.group(1))
            img = Image.open(p).convert("L").resize((G, G), Image.BILINEAR)
            f = (np.asarray(img, np.uint8) > 127).astype(np.uint8).ravel()
            frames.append(f)
            acts.append(sum(1 << k for k, (_, bit) in enumerate(KEYS) if a & bit))
        if (i + 1) % 10 == 0:
            print("已处理 %d/%d 局，%d 帧，用时 %.0fs"
                  % (i + 1, len(dirs), len(frames), time.time() - t0), flush=True)
        if len(frames) and len(frames) % every < len(fs):
            pass
    np.savez_compressed(OUT, frames=np.stack(frames), actions=np.array(acts, np.uint8))
    print("完成：%d 帧 → %s（%.1f MB）"
          % (len(frames), OUT, os.path.getsize(OUT) / 1e6), flush=True)


if __name__ == "__main__":
    main()
