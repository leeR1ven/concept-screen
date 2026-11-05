import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("i-1.gpushare.com", port=21409, username="root",
          password="tYqvm2XBBS4EwkaCTfcSdY7yKeaPEwhg", timeout=30)
cmds = [
    "cd /root/ana && grep -c Traceback walkfar.log; grep -A3 'Error' walkfar.log | tail -12",
    "free -g | head -2; dmesg 2>/dev/null | tail -3",
    "cd /root/ana && tail -2 walkfar.log",
]
for cmd in cmds:
    _, o, e = c.exec_command(cmd, timeout=180)
    print(o.read().decode("utf-8", "replace").strip()[:600])
    print("----")
c.close()
