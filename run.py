"""从 JSON 配置运行 PRC 原始训练、微调或推理入口。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    """将配置转换为明确的命令行参数，固定项目根目录并保留退出码。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="JSON 配置路径")
    parser.add_argument("--dry-run", action="store_true", help="只显示命令，不运行")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tasks = {
        "train": "train_forecasting_transformer.py",
        "infer": "infer_forecasting_DL.py",
        "finetune": "finetune_forecasting_transformer.py",
    }
    command = [sys.executable, "-X", "utf8", str(
        root / "scripts/training" / tasks[config["task"]])]
    for key, value in config["args"].items():
        if value is True:
            command.append("--" + key)
        elif value is False:
            raise ValueError("布尔 false 请直接省略，或按原入口的 --no-* 参数明确指定。")
        elif value is not None:
            command.extend(["--" + key, str(value)])
    print(subprocess.list2cmdline(command), flush=True)
    if args.dry_run:
        return
    env = os.environ.copy()
    temporary = root / "outputs/tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    env.update(TEMP=str(temporary), TMP=str(temporary), MPLBACKEND="Agg",
               PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    env.setdefault("WANDB_MODE", "disabled")
    raise SystemExit(subprocess.call(command, cwd=root, env=env))


if __name__ == "__main__":
    main()
