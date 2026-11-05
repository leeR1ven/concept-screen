"""大规模预训练：用尽可能多的真实马里奥帧，学"下一拍画面"（Wnext）。

架构依据：最后一层这一刻的活动 → 下一刻第一层（学"接下来是什么"）。
判据：预测下一帧的"底部一行"准确率，对比"永远猜上一帧"的基线。
"""
from __future__ import annotations

import glob
import os
import re
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.p51_screen2key import N_IN, PFC, encode  # noqa: E402

ROOT = r"F:\一种AGI关联神经元架构\data\smb_all"
G = 8


def episodes(limit=None, only_win=True):
    dirs = sorted(glob.glob(os.path.join(ROOT, "*")))
    if only_win:
        dirs = [d for d in dirs if "win" in os.path.basename(d).lower()]
    return dirs[:limit] if limit else dirs


def frames_of(d, stride=1):
    fs = sorted(glob.glob(os.path.join(d, "*.png")),
                key=lambda p: int(re.search(r"_f(\d+)_", p).group(1)))
    for p in fs[::stride]:
        img = Image.open(p).convert("L").resize((G, G), Image.BILINEAR)
        yield (np.asarray(img, np.float32) / 255.0 > 0.5).astype(np.float32)


def main(n_episodes=120, stride=2, lr=0.05, report_every=20000):
    pfc = PFC(N_IN)
    Wnext = np.zeros((pfc.n4, N_IN), np.float32)
    u_h = np.zeros(pfc.n4, np.float32)
    u_x = np.zeros(N_IN, np.float32)
    eps = episodes(n_episodes)
    print("episode 数:", len(eps), flush=True)
    n = 0
    t0 = time.time()
    hit = tot = 0
    prev_h = None
    for d in eps:
        prev = None
        fs = list(frames_of(d, stride))
        for f in fs[:-30]:          # 丢掉最后 30 帧，避免临界状态
            x = encode(f)
            h = pfc.step(x, learn=True)
            if prev_h is not None:
                # 学 h_t → x_{t+1}（协方差 + 凹形增长）
                a = prev_h - u_h
                tgt = x - u_x
                phi = 1.0 / (1.0 + np.abs(Wnext))
                Wnext += lr * np.outer(a, tgt) * phi
                np.clip(Wnext, -1.0, 1.0, out=Wnext)
                # 判据：预测"底部一行"（坑/地面所在），与真值比
                pred = prev_h @ Wnext
                bottom = slice((G - 1) * G * 8, G * G * 8)
                truth = x[bottom]
                got = pred[bottom]
                hit += int((truth > 0.5).sum() > 0 and
                           np.argmax(np.abs(got)) == np.argmax(np.abs(truth)))
                tot += 1
            u_h += 0.005 * (h - u_h)
            u_x += 0.005 * (x - u_x)
            prev_h = h
            n += 1
            if n % report_every == 0:
                print("帧 %6d  已用 %.0fs  底部行命中 %.3f"
                      % (n, time.time() - t0, hit / max(1, tot)), flush=True)
    print("完成：%d 帧，用时 %.0fs，底部行命中 %.3f"
          % (n, time.time() - t0, hit / max(1, tot)))
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "Wnext_mario.npy"), Wnext)


if __name__ == "__main__":
    main()
