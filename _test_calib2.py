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
img = _ball(sp, 0.85)

# 用**真实任务输入**校准
_orig = calibrate_force.__defaults__
import agi.calibrate as cal
src_code = cal.calibrate_force.__code__
# 简化：自己跑校准，输入用 img
n_force = int(sp.n_force)
cells = [np.asarray(brain.motor["force"][m], dtype=np.int64) for m in range(sp.n_muscle)]
allc = np.concatenate(cells)
T = 120
drv = np.zeros((T, allc.size), np.float32)
for t in range(T):
    eng.clear_external()
    eng.set_input(brain.encoder.encode(_obs(sp, img)))
    eng.step()
    act = np.asarray(eng.act)
    s_, d_, w_ = np.asarray(eng.src), np.asarray(eng.dst), np.asarray(eng.wp)
    for k, c in enumerate(allc):
        m = d_ == c
        drv[t, k] = float((act[s_[m]] * w_[m]).sum())
thr = np.zeros(allc.size, np.float32)
for k in range(allc.size):
    q = 5 + 65 * (k % n_force) / max(1, n_force - 1)
    thr[k] = float(np.percentile(drv[:, k], q) + 1e-4)
eng.set_thr(allc, thr=thr, soft=np.full(allc.size, 0.60, np.float32))
print("校准门槛范围: %.3f ~ %.3f  驱动均值 %.3f" % (thr.min(), thr.max(), drv.mean()))

for _ in range(30):
    eng.clear_external()
    eng.set_input(brain.encoder.encode(_obs(sp, img)))
    eng.step()
f = np.asarray(brain.forces())
fm = np.asarray(brain.force_mean())
print("forces:", np.round(f, 2), "std=%.3f" % float(f.std()))
print("force_mean:", np.round(fm, 2), "std=%.3f" % float(fm.std()))
