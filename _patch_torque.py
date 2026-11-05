p = "agi/body_go2.py"
s = open(p, encoding="utf-8").read()
old = """        target = self.q_stand + self.posture_offset * f
        tau = self.kp * (target - q) - self.kd * qd"""
new = """        # 两种驱动方式：
        #   位置模式（默认）：大脑给"姿势偏移"，PD 把它变成扭矩；
        #   力矩模式（ANA_TORQUE=1）：大脑**直接给扭矩**，不再经过 PD 位置环。
        # 位置模式里"力度"其实是角度，这才是速度上不去的原因之一。
        if self.torque_mode:
            tau = (2.0 * f - 1.0) * self.torque_max
        else:
            target = self.q_stand + self.posture_offset * f
            tau = self.kp * (target - q) - self.kd * qd"""
assert old in s, "anchor1"
s = s.replace(old, new, 1)
s = s.replace("        self.forces = np.zeros(12, np.float32)",
              "        self.forces = np.zeros(12, np.float32)\n"
              "        self.torque_mode = os.environ.get(\"ANA_TORQUE\", \"\") in (\"1\", \"true\")\n"
              "        self.torque_max = float(os.environ.get(\"ANA_TORQUE_NM\", \"25\"))")
open(p, "w", encoding="utf-8").write(s)
print("torque mode added")
