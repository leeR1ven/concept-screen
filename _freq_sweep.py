import json
import sys

sys.path.insert(0, "/root/ana")
from experiments.p49_walk_far import walk

base = json.load(open("/root/ana/walk_far_nofall.json", encoding="utf-8"))["kw"]
print("cpg_inh cpg_gain | m/s  距离 拍数 倒")
for inh in (0.1, 0.2, 0.35, 0.6):
    for gain in (3.0, 6.0):
        kw = dict(base)
        kw.update(cpg_inh=inh, cpg_gain=gain)
        r = walk(kw)
        print(" %5.2f %6.1f | %.3f %.2f %4d %s"
              % (inh, gain, r["dx"] / max(1, r["ticks"]) * 50.0,
                 r["dx"], r["ticks"], r["fell"]))
