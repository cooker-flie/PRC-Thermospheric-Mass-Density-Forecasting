"""
finetune_forecasting_transformer.py
=====================================================================
增量微调脚本（补充新数据训练，无需从头重新训练）

功能：
  1. 加载已有 checkpoint（如 patchtst_swarm_c_..._2014-01-01_to_2024-12-31_best.pth）
  2. 在更大/更新的时间范围上（如 2014-01-01 ~ 2025-12-31）用小学习率继续训练少量 epoch
  3. 复用主训练脚本的模型结构 / 数据集 / 训练循环 / 验证逻辑，保证与训练完全一致
  4. 沿用旧 checkpoint 的归一化边界（物理边界，与数据范围无关）与特征列

要点：
  - 微调学习率建议远小于原训练 lr（默认 1e-4）
  - 只跑少量 epoch（默认 10），每 epoch 耗时与原训练基本一致（数据量仅增加 2025 一年）
  - 新模型保存为 {model_type}_{satellite}_fold0_ds{ds}_sl{sl}_pl{pl}_{start_date}_to_{end_date}_best.pth，
    不会覆盖原 checkpoint

用法示例：
  python scripts/training/finetune_forecasting_transformer.py \
      --resume models/using/patchtst_swarm_c_fold0_ds1_sl288_pl288_2014-01-01_to_2024-12-31_best.pth \
      --start_date 2014-01-01 --end_date 2025-12-31 \
      --lr 1e-4 --epochs 10 \
      --save_dir models/finetune
=====================================================================
"""

import os
import sys
import argparse
import copy
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader

# 保证能从本目录导入主训练脚本的模型类与数据集
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from train_forecasting_transformer import (
    SatelliteDatasetMultiStep,
    train_single_fold,
    InformerDensityPredictor,
    LSTMMultiStep,
    TransformerMultiStep,
    TemporalFusionTransformer,
    PatchTST,
)


def build_model(model_type, model_kwargs, args):
    """
    按模型类型创建模型实例（与主训练脚本完全一致的分支）。

    Args:
        model_type: 模型类型（informer/lstm/transformer/tft/patchtst）
        model_kwargs: 通用参数（input_size/future_cond_size/d_model/n_heads/dropout/pred_len）
        args: 训练配置（含各模型专属超参）

    Returns:
        nn.Module: 模型实例
    """
    if model_type == 'informer':
        return InformerDensityPredictor(
            **model_kwargs,
            e_layers=args.e_layers, d_ff=args.d_ff, factor=args.factor, d_layers=args.d_layers)
    elif model_type == 'lstm':
        return LSTMMultiStep(model_kwargs['input_size'], model_kwargs['future_cond_size'],
                             args.lstm_hidden, args.lstm_layers, args.dropout, args.pred_len)
    elif model_type == 'transformer':
        return TransformerMultiStep(model_kwargs['input_size'], model_kwargs['future_cond_size'],
                                    args.d_model, args.n_heads, args.e_layers, args.dropout, args.pred_len)
    elif model_type == 'tft':
        return TemporalFusionTransformer(
            input_size=model_kwargs['input_size'],
            future_cond_size=model_kwargs['future_cond_size'],
            seq_len=args.seq_len, pred_len=args.pred_len,
            d_model=args.d_model, n_heads=args.n_heads, num_layers=args.e_layers,
            dropout=args.dropout, lstm_hidden=args.lstm_hidden, lstm_layers=args.lstm_layers,
            use_variable_selection=args.use_variable_selection)
    elif model_type == 'patchtst':
        return PatchTST(
            input_size=model_kwargs['input_size'],
            future_cond_size=model_kwargs['future_cond_size'],
            seq_len=args.seq_len, pred_len=args.pred_len,
            patch_len=args.patch_len, stride=args.patch_stride,
            d_model=args.d_model, n_heads=args.n_heads, num_layers=args.e_layers,
            dropout=args.dropout)
    else:
        raise ValueError(f"Unknown model: {model_type}")


def main():
    parser = argparse.ArgumentParser(
        description='增量微调：加载已有 checkpoint 在更大时间范围上低学习率继续训练',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--resume', type=str,
                        default='models/using/patchtst_swarm_c_fold0_ds1_sl288_pl288_2014-01-01_to_2024-12-31_best.pth',
                        help='已有 checkpoint 路径（作为初始权重）')
    parser.add_argument('--data_path', type=str,
                        default='data/merged_datasets/satellites_data_swarm_c_10xresample_omni.csv',
                        help='训练数据 CSV（默认与主训练一致）')
    parser.add_argument('--satellite', type=str, default='swarm_c', help='卫星（与 checkpoint 一致）')
    parser.add_argument('--start_date', type=str, default='2014-01-01', help='微调数据起始（含）')
    parser.add_argument('--end_date', type=str, default='2025-12-31', help='微调数据结束（含，补充的年份放这里）')
    parser.add_argument('--lr', type=float, default=1e-4, help='微调学习率（远小于原训练 lr）')
    parser.add_argument('--epochs', type=int, default=10, help='微调轮数（少量即可）')
    parser.add_argument('--batch_size', type=int, default=None, help='批量大小（默认沿用 checkpoint）')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='按时间顺序末尾划分的验证集比例')
    parser.add_argument('--save_dir', type=str, default='models/finetune', help='微调模型保存目录')
    parser.add_argument('--device', type=str, default='cuda', help="计算设备：'auto'/'cuda'/'cpu'")
    args = parser.parse_args()

    device = torch.device(
        'cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    print(f"[设备] {device}")

    # ---------- 1. 加载旧 checkpoint ----------
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    old_args = ckpt['args']  # dict
    print(f"[加载] 旧模型: {args.resume}")
    print(f"  模型类型: {old_args.get('model_type')}, seq_len={old_args.get('seq_len')}, "
          f"pred_len={old_args.get('pred_len')}, train_type={old_args.get('train_type')}")
    print(f"  特征列数: {len(ckpt.get('feature_columns', []))}, 未来条件列数: {len(old_args.get('future_condition_columns', []))}")

    # ---------- 2. 构建数据集（新时间范围） ----------
    ft_args = SimpleNamespace(**copy.deepcopy(old_args))
    ft_args.start_date = args.start_date
    ft_args.end_date = args.end_date
    ft_args.satellite = args.satellite
    ft_args.lr = args.lr
    ft_args.epochs = args.epochs
    ft_args.save_dir = args.save_dir
    if args.batch_size:
        ft_args.batch_size = args.batch_size
    # 数据划分验证集比例（覆盖旧 args 中的划分配置）
    ft_args.val_ratio = args.val_ratio

    print(f"\n[数据] 加载数据集: {args.start_date} ~ {args.end_date}（downsample={ft_args.downsample}）")
    dataset = SatelliteDatasetMultiStep(
        csv_path=args.data_path,
        satellite=args.satellite,
        start_date=args.start_date,
        end_date=args.end_date,
        seq_len=ft_args.seq_len,
        pred_len=ft_args.pred_len,
        downsample=ft_args.downsample,
        normalize=True,
    )

    # ---------- 3. 校验特征列与归一化边界 ----------
    ckpt_feats = ckpt.get('feature_columns', [])
    if list(ckpt_feats) != list(dataset.effective_feature_columns):
        raise ValueError(
            f"特征列不一致：checkpoint 有 {len(ckpt_feats)} 列，新数据有 {len(dataset.effective_feature_columns)} 列。\n"
            f"旧: {ckpt_feats}\n新: {dataset.effective_feature_columns}\n"
            f"请确认 --data_path/--satellite 与旧训练一致。")
    print(f"[校验] 特征列一致（{len(dataset.effective_feature_columns)} 列）")
    # 归一化边界为固定物理边界（与数据范围无关），沿用 checkpoint 保存值以防 PHYSICAL_BOUNDS 变动
    for k in ('feature_min', 'feature_max', 'feature_range', 'target_log_min', 'target_log_max', 'target_log_range'):
        if k in ckpt and hasattr(dataset, k):
            setattr(dataset, k, ckpt[k])
    print("[归一化] 沿用 checkpoint 的物理边界归一化参数")

    # ---------- 4. 创建模型并加载初始权重 ----------
    effective_input_size = len(dataset.effective_feature_columns)
    model_kwargs = {
        'input_size': effective_input_size,
        'future_cond_size': len(dataset.future_condition_columns),
        'd_model': ft_args.d_model,
        'n_heads': ft_args.n_heads,
        'dropout': ft_args.dropout,
        'pred_len': ft_args.pred_len,
    }
    model = build_model(ft_args.model_type, model_kwargs, ft_args).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"[模型] 已创建 {ft_args.model_type} 并加载 checkpoint 权重")

    # ---------- 5. 按时间顺序划分 train/val ----------
    n = len(dataset)
    val_size = int(n * args.val_ratio)
    train_indices = list(range(0, n - val_size))
    val_indices = list(range(n - val_size, n))
    train_loader = DataLoader(Subset(dataset, train_indices),
                              batch_size=ft_args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices),
                            batch_size=ft_args.batch_size, shuffle=False)
    print(f"[划分] train={len(train_indices)}, val={len(val_indices)}（时间顺序末尾 {args.val_ratio:.0%}）")

    criterion = nn.MSELoss()

    # ---------- 6. 微调训练（复用主训练脚本 train_single_fold） ----------
    print(f"\n[微调] 学习率={ft_args.lr}, epochs={ft_args.epochs}, batch_size={ft_args.batch_size}")
    train_single_fold(model, train_loader, val_loader, None, criterion, device,
                      ft_args, dataset, fold_idx=0, model_kwargs=model_kwargs)

    # ---------- 7. 提示新模型位置 ----------
    date_str = f"{args.start_date}_to_{args.end_date}"
    save_path = os.path.join(
        args.save_dir,
        f"{ft_args.model_type}_{args.satellite}_fold0_ds{ft_args.downsample}_"
        f"sl{ft_args.seq_len}_pl{ft_args.pred_len}_{date_str}_best.pth")
    print(f"\n[完成] 微调模型: {save_path}")
    print("将该模型路径用于 infer_forecasting_DL.py 的 --model_path 即可。")


if __name__ == '__main__':
    main()
