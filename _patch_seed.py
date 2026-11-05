p = "experiments/p49_walk_far.py"
s = open(p, encoding="utf-8").read()

# 1) 种子优先用"不摔最好"的存档，而不是 champion
old = """    seed = json.load(open(os.path.join(ROOT, "champion.json"), encoding="utf-8"))["kw"]"""
new = """    # 种子优先用"不摔最好"的存档：否则每次重启都从旧种子从头爬，历史成绩白丢
    seed_path = os.path.join(ROOT, "walk_far_nofall.json")
    if not os.path.exists(seed_path):
        seed_path = os.path.join(ROOT, "champion.json")
    seed = json.load(open(seed_path, encoding="utf-8"))["kw"]
    print("种子来自:", os.path.basename(seed_path), flush=True)"""
assert old in s, "anchor seed"
s = s.replace(old, new, 1)

# 2) 破纪录时同时存版本化文件
old2 = """                json.dump({"dx": best_nf[1], "fell": False, "kw": best_nf[0]},
                          f, ensure_ascii=False, indent=2)"""
new2 = """                json.dump({"dx": best_nf[1], "fell": False, "kw": best_nf[0]},
                          f, ensure_ascii=False, indent=2)
            # 版本化：破纪录就单独存一份，避免下次被覆盖后找不回来
            v = os.path.join(ROOT, "walk_nf_%05.3f.json" % best_nf[1])
            with open(v, "w", encoding="utf-8") as f:
                json.dump({"dx": best_nf[1], "fell": False, "kw": best_nf[0]},
                          f, ensure_ascii=False, indent=2)"""
assert old2 in s, "anchor archive"
s = s.replace(old2, new2, 1)
open(p, "w", encoding="utf-8").write(s)
print("seed + versioned archive ok")
