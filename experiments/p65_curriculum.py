"""分层加动作：先只用 2 个动作学会过坑，再逐个加动作（继续训练，不清空）。

判据：加了第 3 个动作后，最远距离仍要 > 1000（一次给 3 个只有 34）。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.p51_screen2key import N_IN, PFC, encode  # noqa: E402
from experiments.p57_play import World  # noqa: E402

N_KEY = 6
A_RUN = [1, 0, 0, 1, 0, 0]
A_RUN_J = [1, 0, 1, 1, 0, 0]
A_JUMP = [0, 0, 1, 1, 0, 0]
A_IDLE = [0, 0, 0, 0, 0, 0]
STAGES = [("2动作", [A_RUN, A_RUN_J]),
          ("3动作", [A_RUN, A_RUN_J, A_JUMP]),
          ("4动作", [A_RUN, A_RUN_J, A_JUMP, A_IDLE])]


class Player:
    def __init__(self, seed=0, lr=0.4, explore=0.4):
        self.pfc = PFC(N_IN)
        self.Wd = np.zeros((self.pfc.n4, 1), np.float32)
        self.u_h = np.zeros(self.pfc.n4, np.float32)
        self.u_d = np.zeros(1, np.float32)
        self.slots = np.arange(N_IN - N_KEY, N_IN)
        self.lr, self.explore = lr, explore
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.best = 0.0

    def learn(self, hb, danger):
        a = hb - self.u_h
        b = np.array([float(danger)], np.float32) - self.u_d
        self.Wd[:] += self.lr * np.outer(a, b) - 0.002 * self.Wd
        np.clip(self.Wd, -1.0, 1.0, out=self.Wd)

    def train(self, cands, ticks, horizon=2, gate=0.0, random_pits=False):
        cands = np.array(cands, np.float32)
        self.pit_rng = np.random.default_rng(self.seed + 42)
        def pits():
            if not random_pits:
                return ((6, 8), (20, 22), (34, 36))
            ps, x = [], 6
            while x < 60:
                ps.append((x, x + 2)); x += int(self.pit_rng.integers(10, 16))
            return tuple(ps)
        w = World(seed=self.seed, pits=pits(), length=60)
        eps, buf, run_best = 0, [], 0.0
        for t in range(ticks):
            if not w.alive:
                run_best = max(run_best, w.x)
                self.best = max(self.best, w.x)
                eps += 1
                for hb in buf:
                    self.learn(hb, 1.0)
                buf.clear()
                w = World(seed=self.seed + eps, pits=pits(), length=60)
            x = encode(w.frame())
            x[self.slots] = 0.0
            XA = np.repeat(x[None, :], len(cands), 0)
            XA[:, self.slots] = cands
            H = np.array([self.pfc.step(XA[i], learn=False) for i in range(len(cands))])
            risks = (((H - self.u_h) @ self.Wd) - self.u_d[0])[:, 0]
            if self.rng.random() < self.explore:
                act = int(self.rng.integers(len(cands)))
            else:
                # 门控：预测危险不高时，只允许"往前走"的动作（不许原地跳/不动）
                allowed = list(range(len(cands)))
                if gate > 0 and float(risks.min()) < gate:
                    allowed = [i for i, c in enumerate(cands) if c[0] > 0.5] or allowed
                act = min(allowed, key=lambda i: risks[i])
            w.step(cands[act])
            buf.append(H[act].copy())
            if len(buf) > horizon:
                self.learn(buf.pop(0), 0.0)
            self.u_h += 0.005 * (H[act] - self.u_h)
        run_best = max(run_best, w.x)
        self.best = max(self.best, w.x)
        return dict(best=run_best, eps=eps + 1)


if __name__ == "__main__":
    p = Player()
    print("%-8s %10s %6s %10s" % ("阶段", "本段最远", "局数", "累计最远"))
    for name, cands in STAGES:
        r = p.train(cands, ticks=7000)
        print("%-8s %10.1f %6d %10.1f" % (name, r["best"], r["eps"], p.best))
