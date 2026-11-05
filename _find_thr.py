p = "agi/build.py"
s = open(p, encoding="utf-8").read()
old = s[s.index("        # 力度档位：阈值按成员数缩放"): s.index("        K = len(members)")] if "力度档位" in s else ""
# 直接用阈值表达式替换
import re
m = re.search(r"thr=np\.full\(n_force, ([^)]*)\)", s)
print("找到阈值表达式:", m.group(1) if m else "未找到")
