p = "agi/body_go2.py"
s = open(p, encoding="utf-8").read()
old = """        if self.torque_mode:
            tau = (2.0 * f - 1.0) * self.torque_max
        else:"""
new = """        if self.torque_mode:
            # 混合：PD 撑住身体（反馈），大脑叠加力矩（前馈推进）。
            # 纯力矩从零学太难（实测 7 拍就倒），混合同真机做法。
            tau = (self.kp * (self.q_stand - q) - self.kd * qd
                   + (2.0 * f - 1.0) * self.torque_max)
        else:"""
assert old in s, "anchor"
s = s.replace(old, new, 1)
s = s.replace("os.environ.get(\"ANA_TORQUE_NM\", \"25\")",
              "os.environ.get(\"ANA_TORQUE_NM\", \"8\")")
open(p, "w", encoding="utf-8").write(s)
print("hybrid torque mode")
