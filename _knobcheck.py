import sys

import numpy as np

sys.path.insert(0, "/root/ana")
from agi.dog_tasks import _ball, _brain, _obs

base = dict(seed=1, k_near=8, region_links=10, w_lo=0.1, w_hi=0.3,
            motor_fan_in=8, posture_offset=0.4, cpg_units=8, cpg_gain=3.0,
            cpg_inh=0.3, force_fan_in=4)


def run(kw, ticks=60):
    sp, brain = _brain(kw)
    eng = brain.engine
    for _ in range(ticks):
        eng.clear_external()
        eng.set_input(brain.encoder.encode(_obs(sp, _ball(sp, 0.85))))
        eng.step()
    f = brain.forces()
    return float(np.asarray(f).mean()), float(np.asarray(f).std())


print("base            :", ["%.3f" % v for v in run(base)])
for k, v in (("k_near", 35), ("w_hi", 5.0), ("posture_offset", 1.2),
             ("cpg_gain", 12.0), ("motor_fan_in", 60), ("force_fan_in", 0)):
    kw = dict(base)
    kw[k] = v
    print("%-16s:" % k, ["%.3f" % x for x in run(kw)])
