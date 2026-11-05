import paramiko

CHK = r'''import os, sys
sys.path.insert(0, "/root/ana")
print("ANA_TORQUE env =", repr(os.environ.get("ANA_TORQUE")))
from agi.body_go2 import Go2Body
b = Go2Body(posture_offset=0.5)
print("body.torque_mode =", b.torque_mode, " torque_max =", b.torque_max)
'''
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("i-1.gpushare.com", port=21409, username="root",
          password="tYqvm2XBBS4EwkaCTfcSdY7yKeaPEwhg", timeout=30)
s = c.open_sftp()
with s.open("/root/ana/chk_torque.py", "w") as f:
    f.write(CHK)
s.close()
_, o, e = c.exec_command(
    "cd /root/ana && export ANA_GO2_XML=/root/go2models/scene.xml ANA_TORQUE=1 && "
    "python3 chk_torque.py 2>&1 | tail -4", timeout=600)
print(o.read().decode("utf-8", "replace").strip()[-500:])
_, o, e = c.exec_command(
    "cd /root/ana && ps -eo args | grep p49_walk_far | head -2", timeout=120)
print("proc:", o.read().decode("utf-8", "replace").strip()[:300])
c.close()
