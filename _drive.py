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

act = np.asarray(eng.act)
src = np.asarray(eng.src)
dst = np.asarray(eng.dst)
w = np.asarray(eng.wp)
f0 = np.asarray(brain.motor["force"][0], dtype=np.int64).ravel()
print("肌肉0 的力细胞 id:", f0[:12])
print("力细胞阈值:", np.round(np.asarray(eng.thr)[f0], 2))
for c in f0[:3]:
    m = dst == c
    print("细胞%d: 上游 %d 条边, 权重和 %.1f, 实际驱动 %.2f, 激活 %.2f"
          % (c, int(m.sum()), float(w[m].sum()),
             float((act[src[m]] * w[m]).sum()), float(act[c])))
