"""使用同一划分运行五种架构的残差学习与直接回归对照。"""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    """生成并依次执行十组配置；默认预览，显式 --execute 才开始训练。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="完整融合数据 CSV")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = root / "outputs/comparison_configs"
    destination.mkdir(parents=True, exist_ok=True)
    template = json.loads((root / "configs/demo_train.json").read_text(encoding="utf-8"))
    for architecture in ["lstm", "transformer", "informer", "tft", "patchtst"]:
        for objective in ["log_exp_residual", "log_density"]:
            configuration = {"task": "train", "args": dict(template["args"])}
            configuration["args"].update(
                data_path=str(Path(args.data).resolve()), start_date=args.start,
                end_date=args.end, model_type=architecture, train_type=objective,
                d_model=128, batch_size=64, epochs=args.epochs,
                val_ratio=0.2, test_ratio=0.1, test_gap_days=0, device=args.device,
                save_dir=f"outputs/comparison/{architecture}/{objective}")
            path = destination / f"{architecture}_{objective}.json"
            path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")
            command = [sys.executable, str(root / "run.py"), str(path)]
            if not args.execute:
                command.append("--dry-run")
            subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
