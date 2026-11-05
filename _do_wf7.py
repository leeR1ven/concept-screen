import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("i-1.gpushare.com", port=21409, username="root",
          password="tYqvm2XBBS4EwkaCTfcSdY7yKeaPEwhg", timeout=30)
s = c.open_sftp()
s.put("_wf6.sh", "/root/wf6.sh")
s.put("agi/body_go2.py", "/root/ana/agi/body_go2.py")
s.put("experiments/p49_walk_far.py", "/root/ana/experiments/p49_walk_far.py")
s.close()
cmd = ("sed -i '1s/^\\xEF\\xBB\\xBF//' /root/wf6.sh; sed -i 's/\\r$//' /root/wf6.sh; "
       "pkill -f p49_walk_far || true; sleep 2; bash /root/wf6.sh")
_, o, e = c.exec_command(cmd, timeout=2400)
print(o.read().decode("utf-8", "replace").strip()[-700:])
print(e.read().decode("utf-8", "replace").strip()[-200:])
c.close()
