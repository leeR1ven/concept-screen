p = "experiments/p49_walk_far.py"
s = open(p, encoding="utf-8").read()
# 放宽：CPG 增益到 20、姿态偏移到 1.2
s = s.replace('* (1 + 0.4 * scale * rng.uniform(-1, 1)), 0.05, 6.0))',
              '* (1 + 0.6 * scale * rng.uniform(-1, 1)), 0.05, 20.0))')
s = s.replace('out[k] = float(np.clip(v * (1 + 0.25 * scale * rng.uniform(-1, 1)), 0.01, 12.0))',
              'out[k] = float(np.clip(v * (1 + 0.35 * scale * rng.uniform(-1, 1)), 0.01, 30.0))')
s = s.replace('        cpg_gain=float(rng.uniform(0.5, 12.0)),',
              '        cpg_gain=float(rng.uniform(0.5, 20.0)),')
s = s.replace('        posture_offset=float(rng.uniform(0.12, 1.2)),',
              '        posture_offset=float(rng.uniform(0.12, 1.6)),')
s = s.replace('    out["posture_offset"] = float(np.clip(kw.get("posture_offset", 0.4)',
              '    out["posture_offset"] = float(np.clip(kw.get("posture_offset", 0.4)')
open(p, "w", encoding="utf-8").write(s)
print("ranges widened")
