import sys

import numpy as np

sys.path.insert(0, ".")
from agi.body_go2 import Go2Body
from agi.calibrate import calibrate_force
from agi.dog_tasks import _ball, _brain, _obs

kw = dict(seed=1, k_near=8, region_links=10, w_lo=0.1, w_hi=0.3,
          motor_fan_in=8, posture_offset=0.5, cpg_units=8, cpg_gain=3.0,
          cpg_inh=0.3, force_fan_in=4)
sp, brain = _brain(kw)
eng = brain.engine

# 校准前
for _ in range(30):
    eng.clear_external()
    eng.set_input(brain.encoder.encode(_obs(sp, _ball(sp, 0.85))))
    eng.step()
print("校准前 forces:", np.round(np.asarray(brain.forces()), 2),
      "std=%.3f" % float(np.std(brain.forces())))

info = calibrate_force(eng, brain, sp, ticks=40)
print("校准信息:", {k: round(v, 2) for k, v in info.items()})

for _ in range(30):
    eng.clear_external()
    eng.set_input(brain.encoder.encode(_obs(sp, _ball(sp, 0.85))))
    eng.step()
f = np.asarray(brain.force_mean())
print("校准后 force_mean:", np.round(f, 2), "std=%.3f" % float(f.std()))
