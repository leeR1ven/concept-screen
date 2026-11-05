import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("i-1.gpushare.com", port=21409, username="root",
          password="tYqvm2XBBS4EwkaCTfcSdY7yKeaPEwhg", timeout=30)
cmd = ("cd /root/ana && export ANA_GO2_XML=/root/go2models/scene.xml "
       "OMP_NUM_THREADS=1 ANA_TORQUE=1 ANA_TORQUE_NM=25 && "
       "timeout 900 python3 experiments/p49_walk_far.py --gens 1 --pop 8 --jobs 8 "
       "2>&1 | tail -6")
_, o, e = c.exec_command(cmd, timeout=1200)
print("TORQUE MODE 1 gen:")
print(o.read().decode("utf-8", "replace").strip()[-600:])
c.close()
