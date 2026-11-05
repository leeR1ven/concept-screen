import json
import sys

sys.path.insert(0, "/root/ana")
from experiments.p49_walk_far import walk

base = json.load(open("/root/ana/walk_far_nofall.json", encoding="utf-8"))["kw"]
print("posture_offset | m/s  距离 拍数 倒")
for po in (0.4, 0.6, 0.8, 1.0, 1.2, 1.6):
    kw = dict(base)
    kw["posture_offset"] = po
    r = walk(kw)
    print(" %5.1f | %.3f %.2f %4d %s"
          % (po, r["dx"] / max(1, r["ticks"]) * 50.0, r["dx"], r["ticks"], r["fell"]))
