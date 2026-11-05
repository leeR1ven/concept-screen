import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("i-1.gpushare.com", port=21409, username="root",
          password="tYqvm2XBBS4EwkaCTfcSdY7yKeaPEwhg", timeout=30)
script = """#!/bin/bash
# 看门狗（修正版）：只有进程数为 0、且日志 15 分钟没更新，才认为真的死了。
cd /root/ana || exit 1
while true; do
  n=$(pgrep -fc p49_walk_far)
  if [ "$n" -eq 0 ]; then
    age=$(( $(date +%s) - $(stat -c %Y walkfar.log 2>/dev/null || echo 0) ))
    if [ "$age" -gt 900 ]; then
      echo "[$(date '+%H:%M:%S')] 真的停了(age=${age}s)，重启" >> watchdog.log
      bash /root/wf5.sh >> /dev/null 2>&1
    else
      echo "[$(date '+%H:%M:%S')] 进程0但日志新鲜(${age}s)，先观察" >> watchdog.log
    fi
  fi
  sleep 300
done
"""
sftp = c.open_sftp()
with sftp.open("/root/watchdog.sh", "w") as f:
    f.write(script)
sftp.close()
cmd = ("pkill -f watchdog.sh || true; sleep 1; chmod +x /root/watchdog.sh; "
       "setsid nohup bash /root/watchdog.sh > /dev/null 2>&1 < /dev/null & "
       "sleep 10; pgrep -fc watchdog.sh; pgrep -fc p49_walk_far; "
       "tail -2 /root/ana/watchdog.log")
_, o, e = c.exec_command(cmd, timeout=600)
print(o.read().decode("utf-8", "replace").strip()[-400:])
c.close()
