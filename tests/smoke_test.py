"""使用真实数据验证加载、单批次训练及已有权重的残差重建推理。"""

import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TEMPORARY = ROOT / "outputs/tmp"
TEMPORARY.mkdir(parents=True, exist_ok=True)
os.environ.update(TEMP=str(TEMPORARY), TMP=str(TEMPORARY), WANDB_MODE="disabled",
                  MPLBACKEND="Agg", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
sys.path.insert(0, str(ROOT / "scripts/training"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from infer_forecasting_DL import ForecastingPredictor  # noqa: E402
from train_forecasting_transformer import PatchTST, SatelliteDatasetMultiStep  # noqa: E402


def main():
    """检查真实张量、梯度和密度输出；不宣称复现完整论文实验。"""
    torch.set_num_threads(2)
    data = ROOT / "data/merged_datasets"
    dataset = SatelliteDatasetMultiStep(
        str(data / "satellites_data_swarm_c_10xresample_omni.csv"),
        "swarm_c", "2024-05-01", "2024-05-31", seq_len=288,
        pred_len=288, downsample=1)
    sample = dataset[0]
    x = sample["features"].unsqueeze(0)
    assert x.shape == (1, 288, 44), x.shape
    assert torch.isfinite(x).all()
    model = PatchTST(input_size=44, future_cond_size=0, seq_len=288,
                     pred_len=288, d_model=64, n_heads=4, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    prediction = model(x)
    target = sample["target"].unsqueeze(0)
    baseline = sample["nrlmsise_baseline"].unsqueeze(0)
    loss = torch.nn.functional.mse_loss(prediction + baseline, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    weight = next((ROOT / "models/finetune").glob("*.pth"))
    predictor = ForecastingPredictor(str(weight), device="cpu")
    frame = pd.read_csv(data / "satellites_data_swarm_a_10xresample_omni.csv", nrows=577)
    densities, indices = predictor.predict_from_dataframe(frame)
    assert densities.shape == (2, 288), densities.shape
    assert np.isfinite(densities).all() and (densities > 0).all()
    result = {"status": "passed", "torch": torch.__version__,
              "numpy": np.__version__, "pandas": pd.__version__,
              "features": 44, "training_batch_loss": float(loss.detach()),
              "inference_shape": list(densities.shape),
              "checkpoint_d_model": predictor.args["d_model"],
              "scope": "真实数据单批次训练和主权重推理；非完整论文重训"}
    (ROOT / "outputs/smoke_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
