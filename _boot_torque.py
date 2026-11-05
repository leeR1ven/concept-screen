import paramiko

LOCAL = r"F:\一种AGI关联神经元架构\ana"
SCRIPT = """#!/bin/bash
cd /root/ana || exit 1
export ANA_TORQUE=1
export ANA_TORQUE_NM=25
export OMP_NUM_THREADS=1
export ANA_GO2_XML=/root/go2models/scene.xml
python3 experiments/p49_walk_far.py --gens 60 --pop 40 --jobs 28 > /root/ana/walkfar.log 2>&1
"""

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("i-1.gpushare.com", port=21409, username="root",
          password="tYqvm2XBBS4EwkaCTfcSdY7yKeaPEwhg", timeout=30)
s = c.open_sftp()
s.put(LOCAL + r"\agi\body_go2.py", "/root/ana/agi/body_go2.py")
s.put(LOCAL + r"\experiments\p49_walk_far.py", "/root/ana/experiments/p49_walk_far.py")
with s.open("/root/ana/torque_run.sh", "w") as f:
    f.write(SCRIPT)
s.close()

setup = ("pkill -f p49_walk_far || true; pkill -f watchdog.sh || true; sleep 2; "
         "chmod +x /root/ana/torque_run.sh; : > /root/ana/walkfar.log; "
         "setsid nohup bash /root/ana/torque_run.sh > /dev/null 2>&1 < /dev/null & ")
_, o, e = c.exec_command(setup, timeout=120)
o.read()
_, o, e = c.exec_command("sleep 180; pgrep -fc p49_walk_far; "
                         "grep -c '第' /root/ana/walkfar.log; "
                         "tail -3 /root/ana/walkfar.log", timeout=900)
print(o.read().decode("utf-8", "replace").strip()[-800:])
print(e.read().decode("utf-8", "replace").strip()[-200:])
c.close()
