"""跑 N 轮训练（固定入口），进度直接打印，便于看曲线。

用法：python experiments/p79_run200.py 40
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.p78_batched import main  # noqa: E402

if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    main(epochs=epochs)
