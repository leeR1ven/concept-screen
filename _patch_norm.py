p = "agi/build.py"
s = open(p, encoding="utf-8").read()
old = """        eng.connect(np.repeat(f, K), np.tile(members, spec.n_force),
                    np.ones(spec.n_force * K, np.float32), group=G.MOTOR_DRIVE)"""
new = """        # 权重 = 1/K：驱动变成"成员里活跃的比例"(0~1)，与阈值同一量纲。
        # 架构说门槛要按该神经元自己的信号来定——归一化后这个比例就是信号本身。
        eng.connect(np.repeat(f, K), np.tile(members, spec.n_force),
                    np.full(spec.n_force * K, 1.0 / float(K), np.float32),
                    group=G.MOTOR_DRIVE)"""
assert old in s, "anchor weight"
s = s.replace(old, new, 1)

old2 = """        _lo = float(getattr(spec, "force_thr_lo", 0.50))
        _hi = float(getattr(spec, "force_thr_hi", 1.00))
        _frac = _lo + (_hi - _lo) * np.arange(spec.n_force, dtype=np.float32) / max(
            1, spec.n_force - 1)
        thr = (float(K) * _frac).astype(np.float32)"""
new2 = """        # 阈值直接是"活跃比例"：第 0 档 0.30，最后一档 0.95（成员几乎全亮）
        _lo = float(getattr(spec, "force_thr_lo", 0.30))
        _hi = float(getattr(spec, "force_thr_hi", 0.95))
        thr = (_lo + (_hi - _lo) * np.arange(spec.n_force, dtype=np.float32) / max(
            1, spec.n_force - 1)).astype(np.float32)"""
assert old2 in s, "anchor thr"
s = s.replace(old2, new2, 1)
open(p, "w", encoding="utf-8").write(s)
print("normalized weights + ratio thresholds")
