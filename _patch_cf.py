p = "experiments/p49_walk_far.py"
s = open(p, encoding="utf-8").read()
old = "        body.step(brain.forces())"
new = ("        # 用**连续**力度（力细胞的平均激活）而不是\"亮了几档\"。\n"
       "        # 计数式在多数细胞同时亮/灭时会退化成 0/1 开关，实测 std=0.000，\n"
       "        # 等于没有分级控制——这是\"参数怎么变结果都一样\"的根因。\n"
       "        body.step(brain.force_mean())")
assert old in s, "anchor"
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("walk uses continuous force")
