"""检查发布文件语法、入口导入和配置命令，禁止悄悄依赖旧项目。"""

import ast
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def check_entry(name):
    """执行只读帮助检查，不联网下载或训练。"""
    result = subprocess.run([sys.executable, "-X", "utf8", name, "--help"],
                            cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=120)
    if result.returncode:
        raise RuntimeError(name + "\n" + result.stdout[-1500:] + result.stderr[-1500:])
    return name


def main():
    """验证所有 Python 语法以及保留的训练、微调、推理、预处理入口。"""
    temporary = ROOT / "outputs/tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ.update(TEMP=str(temporary), TMP=str(temporary), WANDB_MODE="disabled",
                      MPLBACKEND="Agg", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    paths = list(ROOT.rglob("*.py"))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        ast.parse(text, filename=str(path))
        assert ("E:/working/" + "karman") not in text
    entries = [
        "scripts/training/train_forecasting_transformer.py",
        "scripts/training/finetune_forecasting_transformer.py",
        "scripts/training/infer_forecasting_DL.py",
        "scripts/training/infer_plotting.py",
        "scripts/input_data_prep/process_merge_sw_and_tudelft_data_simple.py",
        "scripts/input_data_prep/download_tudelft_thermo.py",
        "scripts/input_data_prep/download_omniweb.py",
        "experiments/compare_architectures.py",
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        checked = list(executor.map(check_entry, entries))
    for name in ["demo_train", "demo_infer"]:
        subprocess.run([sys.executable, "run.py", f"configs/{name}.json", "--dry-run"],
                       cwd=ROOT, check=True, capture_output=True)
    print(json.dumps({"status": "passed", "python_files": len(paths),
                      "entrypoints": checked, "config_dry_runs": 2}, indent=2))


if __name__ == "__main__":
    main()
