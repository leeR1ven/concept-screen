p = "agi/build.py"
s = open(p, encoding="utf-8").read()
old = """        thr = (float(K) * (0.10 + 0.09 * np.arange(spec.n_force,
                                                   dtype=np.float32))).astype(np.float32)"""
new = """        # 温度计的量程要覆盖**实际驱动**。实测肌肉成员约 80% 亮着，
        # 而旧阈值只到 0.91 倍成员数，导致 10 档几乎全过（力度恒 1）。
        # 现在阈值铺满 0.5~1.0 倍成员数：0.8 的驱动落在中段 → 亮 6 档左右，形成分级。
        _lo = float(getattr(spec, "force_thr_lo", 0.50))
        _hi = float(getattr(spec, "force_thr_hi", 1.00))
        _frac = _lo + (_hi - _lo) * np.arange(spec.n_force, dtype=np.float32) / max(
            1, spec.n_force - 1)
        thr = (float(K) * _frac).astype(np.float32)"""
assert old in s, "anchor"
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)

p2 = "agi/spec.py"
t = open(p2, encoding="utf-8").read()
if "force_thr_lo" not in t:
    t = t.replace("    force_fan_in: int = 0",
                  "    force_thr_lo: float = 0.50                      # 第 1 档阈值（占成员数比例）\n"
                  "    force_thr_hi: float = 1.00                      # 最后一档阈值\n"
                  "    force_fan_in: int = 0")
    open(p2, "w", encoding="utf-8").write(t)
print("force thermometer recalibrated")
