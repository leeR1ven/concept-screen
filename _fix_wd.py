import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("i-1.gpushare.com", port=21409, username="root",
          password="tYqvm2XBBS4EwkaCTfcSdY7yKeaPEwhg", timeout=30)
_, o, e = c.exec_command("pgrep -fc watchdog.sh; tail -4 /root/ana/watchdog.log", timeout=120)
print("watchdog:", o.read().decode("utf-8", "replace").strip()[:400])
cmd = ("bash /root/wf5.sh; "
       "pgrep -fc watchdog.sh")
_, o, e = c.exec_command(cmd, timeout=900)
print("after restart:", o.read().decode("utf-8", "replace").strip()[-300:])
c.close()
