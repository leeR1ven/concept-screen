p = "experiments/p49_walk_far.py"
s = open(p, encoding="utf-8").read()
start = s.index("        # 速度 × 贴地时间占比")
end = s.index("        by_nofall = sorted(")
new = """        # 唯一的评分：速度（米/秒）。摔倒的算无效，其余不加任何权重。
        scored = [(kw, r["dx"] / max(1, r["ticks"]) * 50.0 if not r["fell"] else -1.0,
                   r["fell"])
                  for kw, r in outs]

"""
s = s[:start] + new + s[end:]
open(p, "w", encoding="utf-8").write(s)
print("simple scoring: speed only, no-fall required")
