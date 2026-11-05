import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("i-1.gpushare.com", port=21409, username="root",
          password="tYqvm2XBBS4EwkaCTfcSdY7yKeaPEwhg", timeout=30)
setup = ("pkill -f p49_walk_far || true; pkill -f torque_run || true; sleep 3; "
         ": > /root/ana/walkfar.log; "
         "cd /root/ana && setsid nohup env ANA_GO2_XML=/root/go2models/scene.xml "
         "OMP_NUM_THREADS=1 ANA_TORQUE=1 ANA_TORQUE_NM=25 "
         "python3 experiments/p49_walk_far.py --gens 80 --pop 40 --jobs 28 "
         "> /root/ana/walkfar.log 2>&1 < /dev/null & ")
_, o, e = c.exec_command(setup, timeout=180)
o.read()
_, o, e = c.exec_command("sleep 200; pgrep -fc p49_walk_far; "
                         "tail -3 /root/ana/walkfar.log", timeout=900)
print(o.read().decode("utf-8", "replace").strip()[-600:])
c.close()
