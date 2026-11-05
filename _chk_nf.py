import json

import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("i-1.gpushare.com", port=21409, username="root",
          password="tYqvm2XBBS4EwkaCTfcSdY7yKeaPEwhg", timeout=30)
_, o, e = c.exec_command("cd /root/ana && grep 最快 walkfar.log | tail -2; "
                         "echo ---; pgrep -fc p49_walk_far", timeout=300)
print(o.read().decode("utf-8", "replace").strip()[-500:])
s = c.open_sftp()
for n in ("walk_far_nofall.json", "walk_far.json"):
    try:
        with s.open("/root/ana/" + n) as f:
            d = json.load(f)
        print(n, "dx", round(d["dx"], 3), "倒", d["fell"])
        if n == "walk_far_nofall.json":
            with open(r"F:\一种AGI关联神经元架构\ana\walk_far.json", "w",
                      encoding="utf-8") as g:
                json.dump({"kw": d["kw"]}, g, ensure_ascii=False, indent=2)
    except Exception as ex:
        print(n, type(ex).__name__)
s.close()
c.close()
