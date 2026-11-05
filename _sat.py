import sys

import numpy as np

sys.path.insert(0, "/root/ana")
from agi.dog_tasks import _ball, _brain, _obs

kw = dict(seed=1, k_near=8, region_links=10, w_lo=0.1, w_hi=0.3,
          motor_fan_in=8, posture_offset=0.4, cpg_units=8, cpg_gain=3.0,
          cpg_inh=0.3, force_fan_in=4)
sp, brain = _brain(kw)
eng = brain.engine
for _ in range(40):
    eng.clear_external()
    eng.set_input(brain.encoder.encode(_obs(sp, _ball(sp, 0.85))))
    eng.step()
a = np.asarray(eng.act)
mr1 = np.asarray(brain.motor["mr1"], dtype=np.int64).ravel()
print("mr1 单元活跃比例: %.2f（1.00 表示全亮）" % float((a[mr1] > 0.3).mean()))
print("mr1 平均激活: %.3f" % float(a[mr1].mean()))
for m in (0, 3, 9):
    f = np.asarray(brain.motor["force"][m], dtype=np.int64).ravel()
    print("肌%d: 力细胞亮 %d/10, 平均 %.2f" % (m, int((a[f] > 0.3).sum()), float(a[f].mean())))
print("forces() =", np.round(np.asarray(brain.forces()), 2))
print("force_mean() =", np.round(brain.force_mean(), 2))
