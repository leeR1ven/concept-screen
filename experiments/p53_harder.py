"""更难的世界：更快 + 画面噪声 + 两个方块。看准确率掉多少、还能不能学。"""
import sys

import numpy as np

sys.path.insert(0, ".")
import experiments.p51_screen2key as P
from experiments.p51_screen2key import N_BTN, H, W


def make_world(speed=1.0, noise=0.0, two=False):
    st = {"r": 4.0, "c": 3.0, "vr": 0.35 * speed, "vc": 0.30 * speed}
    st2 = {"r": 2.0, "c": 5.0, "vr": -0.25 * speed, "vc": 0.20 * speed}

    def synth(rng):
        def step(s):
            s["r"] += s["vr"]; s["c"] += s["vc"]
            if s["r"] < 0.5 or s["r"] > H - 1.5:
                s["vr"] = -s["vr"]; s["r"] += s["vr"]
            if s["c"] < 0.5 or s["c"] > W - 1.5:
                s["vc"] = -s["vc"]; s["c"] += s["vc"]
            return int(round(s["r"])), int(round(s["c"]))
        f = np.zeros((H, W), np.float32)
        r, c = step(st)
        f[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = 1.0
        if two:
            r2, c2 = step(st2)
            f[max(0, r2 - 1):r2 + 2, max(0, c2 - 1):c2 + 2] = 1.0
        if noise > 0:
            flip = rng.random((H, W)) < noise
            f[flip] = 1.0 - f[flip]
        btn = np.zeros(N_BTN, np.float32)
        btn[c % N_BTN] = 1.0
        return f, btn
    return synth


def run(speed=1.0, noise=0.0, two=False, ticks=8000):
    P.synth_frame = make_world(speed, noise, two)
    P._STATE.update({"r": 4.0, "c": 3.0, "vr": 0.35 * speed, "vc": 0.30 * speed})
    print("world: speed=%.1f noise=%.2f two=%s" % (speed, noise, two))
    P.main(ticks=ticks)


if __name__ == "__main__":
    for kw in (dict(speed=1.0), dict(speed=3.0), dict(speed=3.0, noise=0.10),
               dict(speed=3.0, noise=0.10, two=True)):
        run(**kw)
