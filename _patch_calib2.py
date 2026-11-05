p = r"agi/calibrate.py"
s = open(p, encoding="utf-8").read()
old = """    thr = np.zeros(all_cells.size, np.float32)
    for m in range(sp.n_muscle):
        sl = slice(m * n_force, (m + 1) * n_force)
        d = drive[:, sl].mean(axis=0)          # 每个细胞自己的典型驱动
        lo, hi = float(np.percentile(d, 10)), float(np.percentile(d, 99) + 1e-3)
        thr[sl] = lo + (hi - lo) * (qs - 20) / 75.0"""
new = """    thr = np.zeros(all_cells.size, np.float32)
    for m in range(sp.n_muscle):
        sl = slice(m * n_force, (m + 1) * n_force)
        for k in range(n_force):
            d = drive[:, m * n_force + k]       # 这个细胞自己的驱动序列
            q = 20 + 70 * k / max(1, n_force - 1)
            thr[m * n_force + k] = float(np.percentile(d, q) + 1e-3)"""
assert old in s, "anchor"
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("per-cell thresholds")
