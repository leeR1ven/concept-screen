"""最小原型：画面 → 前额叶（四层，最后一层学"下一拍"）→ 按键。

只照 docs/ARCHITECTURE.md 的规则搭：
  * 输入层温度计正反对编码（每对恰好亮一个）；
  * 中间层固定随机、局部、门槛按自身信号定；
  * 最后一层挑最强 top-m（m = 第一层亮数）；
  * 学习只有一条：时序赫布（协方差形式，THEORY.md §2.2）；
  * 出生回路 G：最后一层能反过来点亮第一层。
先用**合成数据**（下一帧按键是当前画面的确定函数）验证管道能学到规律。
"""
from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(0)
H = W = 8                 # 画面降采样到 8×8
L = 4                     # 每个像素的等级数
NPIX = H * W
N_IN = NPIX * L * 2       # 正反对：每像素 2L 个
N_BTN = 8                 # 8 个按键
MOT_PER_BTN = 6           # 每个按键 6 个力度神经元
N_MOT = N_BTN * MOT_PER_BTN


def encode(frame: np.ndarray) -> np.ndarray:
    """温度计正反对：每个像素亮 L 个（正）或 L 个（反），永远一半亮。"""
    x = np.zeros(N_IN, np.float32)
    for i, v in enumerate(frame.ravel()):
        lv = int(np.clip(v * L, 0, L - 1))
        base = i * 2 * L
        x[base + lv] = 1.0                       # 正：亮第 lv 个
        x[base + L + (L - 1 - lv)] = 1.0      # 反：亮对称那个
    return x


def local_mask(n_out: int, n_in: int, k: int) -> np.ndarray:
    """每个输出只接附近 k 个输入（按一维位置算远近）。"""
    pos_o = np.arange(n_out)[:, None] / max(1, n_out - 1)
    pos_i = np.arange(n_in)[None, :] / max(1, n_in - 1)
    d = np.abs(pos_o - pos_i)
    idx = np.argsort(d, axis=1)[:, :k]
    m = np.zeros((n_out, n_in), np.float32)
    for o in range(n_out):
        m[o, idx[o]] = RNG.uniform(0.5, 1.5, k)
    return m


class PFC:
    def __init__(self, n1: int, n2: int = 1024, n3: int = 1024, n4: int = 2048):
        self.n1, self.n4 = n1, n4
        self.F2 = local_mask(n2, n1, 8)
        self.F3 = local_mask(n3, n2, 8)
        self.F4 = local_mask(n4, n3, 8)
        self.thr2 = np.zeros(n2, np.float32)
        self.thr3 = np.zeros(n3, np.float32)
        # 出生回路 G：最后一层 → 第一层（固定随机、稀疏）
        self.G = (RNG.random((n4, n1)) < 0.02).astype(np.float32) * RNG.uniform(0.3, 1.0, (n4, n1))
        # 学习到的读出 W：最后一层 → 第一层
        self.W = np.zeros((n4, n1), np.float32)
        self.m = None                        # 最后一层亮的数量 = 第一层亮的数量
        self.h = np.zeros(n4, np.float32)
        self._calib = 0

    def _set_thr(self, F, thr, x, rho=0.3):
        d = F @ x
        thr[:] = np.quantile(d, 1.0 - rho)

    def step(self, x: np.ndarray, learn: bool = True):
        # 第一层 = 外部输入（想法这里先不加，保持简单）
        l1 = x
        if self.m is None:
            self.m = max(1, int(np.asarray(l1).sum()))
        if self._calib < 3:
            self._set_thr(self.F2, self.thr2, l1)
            self._calib += 1
        l2 = (self.F2 @ l1 - self.thr2 > 0).astype(np.float32)
        if self._calib < 3:
            self._set_thr(self.F3, self.thr3, l2)
        l3 = (self.F3 @ l2 - self.thr3 > 0).astype(np.float32)
        d4 = self.F4 @ l3
        h = np.zeros(self.n4, np.float32)
        h[np.argsort(d4)[-self.m:]] = 1.0      # 最后一层：挑最强的 m 个
        if learn:
            # 协方差形式：两侧都只记"比平时多"的部分（简单用中心化）
            drive = self.G @ l1 + self.W @ l1
            d = drive - np.mean(drive)
            a = h - np.mean(h)
            self.W += 0.05 * np.outer(a, l1) - 0.001 * self.W
        self.h = h
        return h


_STATE = {"r": 4.0, "c": 3.0, "vr": 0.35, "vc": 0.30}


def synth_frame(rng) -> tuple[np.ndarray, np.ndarray]:
    """连续世界：一个方块平滑移动，碰到边界反弹。按键 = 方块所在列。

    这样下一帧的按键可以从当前画面预测（否则 i.i.d. 随机帧无法预测）。
    """
    st = _STATE
    st["r"] += st["vr"]
    st["c"] += st["vc"]
    if st["r"] < 0.5 or st["r"] > H - 1.5:
        st["vr"] = -st["vr"]; st["r"] += st["vr"]
    if st["c"] < 0.5 or st["c"] > W - 1.5:
        st["vc"] = -st["vc"]; st["c"] += st["vc"]
    f = np.zeros((H, W), np.float32)
    r, c = int(round(st["r"])), int(round(st["c"]))
    f[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = 1.0
    btn = np.zeros(N_BTN, np.float32)
    btn[c % N_BTN] = 1.0
    return f, btn


def main(ticks: int = 8000, lr: float = 0.15):
    """学 h_t → 下一拍按键（协方差形式：两侧都减平时水平，少一起亮的变负）。"""
    pfc = PFC(N_IN)
    # 按键读出：Wb (n4 × N_BTN)，与 W 同一条赫布规则
    Wb = np.zeros((pfc.n4, N_BTN), np.float32)
    u_h = np.zeros(pfc.n4, np.float32)      # 最后一层的平时水平
    u_b = np.zeros(N_BTN, np.float32)       # 按键的平时水平
    hit = tot = 0
    cnt = np.zeros(N_BTN)
    prev_h = None
    prev_btn = np.zeros(N_BTN, np.float32)
    for t in range(ticks):
        f, btn = synth_frame(RNG)
        h = pfc.step(encode(f), learn=True)

        if prev_h is not None:
            # 预测这一拍的按键：用上一拍的 h（这才是"预测下一拍"）
            drive = (prev_h - u_h) @ Wb - (u_b - u_b.mean())
            pred = int(np.argmax(drive))
            if t > ticks // 2:
                tot += 1
                if pred == int(np.argmax(btn)):
                    hit += 1
                cnt[int(np.argmax(btn))] += 1
            # 协方差赫布：两侧都只记"比平时多"的部分
            a = prev_h - u_h
            b = (btn > 0.5).astype(np.float32) - u_b
            Wb += lr * np.outer(a, b) - 0.002 * Wb
            np.clip(Wb, -1.0, 1.0, out=Wb)

        # 平时水平缓慢跟随（时间常数 ~200 拍）
        u_h += 0.005 * (h - u_h)
        u_b += 0.005 * (btn - u_b)
        prev_h = h
        prev_btn = btn
    acc = hit / max(1, tot)
    print("按键预测准确率: %.3f   最常见键基线: %.3f   样本 %d"
          % (acc, float(cnt.max() / max(1, tot)), tot))


if __name__ == "__main__":
    main()
