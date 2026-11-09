"""
infer_forecasting_DL.py
=====================================================================
Forecasting 模型推理脚本（对应当前主线训练脚本 train_forecasting_transformer.py）

功能：
  1. 加载 train_forecasting_transformer.py 保存的 checkpoint（PatchTST 使用预测头，不使用未来条件解码器）
  2. 读取融合后的卫星轨迹 CSV，构造“历史特征 + 未来显式条件”
  3. 输出逐步长（lead step）的多步密度预报结果 CSV
  4. （可选）自动绘图：密度时序对比 + 逐步长 MAPE 柱状图
  5. （可选）逐步长 MAPE/RMSE 汇总，输出 JSON
  6. （可选）分地磁相位分析：quiet ap≤30 / minor 30<ap≤50 / major ap>50
  7. （可选）控制台打印 NN vs NRLMSIS 三相位汇总表

用法示例：
  python scripts/training/infer_forecasting_DL.py \
      --model_path models/temp/informer_swarm_a_xxx_best.pth \
      --trajectory_file data/merged_datasets/satellites_data_swarm_a_10xresample_omni.csv \
      --satellite swarm_a \
      --start_time "2023-05-09 00:00:00" \
      --end_time "2023-05-11 00:00:00" \
      --output results/2023infer \
      --plot all \
      --summary \
      --report
  # 结果（inference_results.csv / summary.json / 图1~图4）自动保存到
  # results/2023infer/<YYYYMMDD_HHMMSS>/ 带时间戳子文件夹中
  # --plot 选项：none=不绘图 / all=图1~图4 / 数字组合如 "1,3"、"134" 只画对应编号的图
=====================================================================
"""

import os
import sys
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# 尝试导入 pymsis；若不可用则跳过无预报值（持久化）NRLMSISE 基线计算
try:
    import pymsis
    PYMSIS_AVAILABLE = True
except ImportError:
    PYMSIS_AVAILABLE = False
    print('Warning: pymsis 不可用，将跳过持久化 NRLMSISE 基线计算（--persist_baseline）')

# 保证能从本目录导入训练脚本中的模型定义与常量
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# 复用训练脚本中的模型类、特征列与归一化边界，确保与训练完全一致
from train_forecasting_transformer import (
    PHYSICAL_BOUNDS,
    FEATURE_COLUMNS,
    FUTURE_CONDITION_COLUMNS,
    TARGET_COLUMN,
    NRLMSISE_COLUMN,
    NRLMSISE00_COLUMN,
    DATETIME_COLUMN,
    SATELLITE_COLUMN,
    InformerDensityPredictor,
    AutoformerDensityPredictor,
    LSTMMultiStep,
    TransformerMultiStep,
    TemporalFusionTransformer,
    PatchTST,
)

# 复用绘图/汇总/报告模块（infer_plotting.py）：推理脚本仅负责产出结果表，绘图统一走该模块
from infer_plotting import (
    parse_plot_flags,
    PHASE_QUIET_MAX,
    PHASE_MINOR_MAX,
    summarize_by_lead,
    summarize_by_phase,
    overall_metrics,
    plot_results,
    print_report,
)

# 默认地磁指数列（用于分相位分析）
AP_COLUMN_DEFAULT = 'celestrack__ap_average__'

# 推理分块前向的 batch 大小：避免逐窗口（batch=1）推理过慢；显存不足时可调小
INFER_BATCH_SIZE = 256


class ForecastingPredictor:
    """加载 forecasting checkpoint 并进行多步密度预报"""

    def __init__(self, model_path, device='cpu'):
        """
        初始化：加载 checkpoint，恢复归一化参数与模型结构

        Args:
            model_path: 训练保存的 .pth 文件路径
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 加载 checkpoint（训练脚本在 torch.save 中保存了完整配置；仅加载可信来源的权重）
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        self.args = checkpoint['args']
        # 训练时实际使用的特征列
        self.feature_columns = checkpoint['feature_columns']
        # 未来显式条件列（优先从 checkpoint 读取，兼容旧模型）
        self.future_condition_columns = self.args.get('future_condition_columns', FUTURE_CONDITION_COLUMNS)

        # 目标（log 密度）归一化参数
        self.target_log_min = checkpoint['target_log_min']
        self.target_log_max = checkpoint['target_log_max']
        self.target_log_range = checkpoint['target_log_range']

        # 特征 Min-Max 物理边界归一化参数
        self.feature_min = checkpoint['feature_min']
        self.feature_range = checkpoint['feature_range']
        self.feature_range[self.feature_range == 0] = 1.0  # 防止除零

        # 从 checkpoint 恢复模型超参数
        self.model_type = self.args.get('model_type', 'informer')
        self.seq_len = self.args.get('seq_len', 288)
        self.pred_len = self.args.get('pred_len', 12)
        self.train_type = self.args.get('train_type', 'log_exp_residual')

        input_size = len(self.feature_columns)
        future_cond_size = len(self.future_condition_columns)

        # ---------- 按模型类型重建结构（必须与训练脚本一致） ----------
        if self.model_type == 'informer':
            self.model = InformerDensityPredictor(
                input_size=input_size,
                future_cond_size=future_cond_size,
                d_model=self.args.get('d_model', 64),
                n_heads=self.args.get('n_heads', 4),
                e_layers=self.args.get('e_layers', 2),
                d_layers=self.args.get('d_layers', 1),
                d_ff=self.args.get('d_ff', 512),
                dropout=self.args.get('dropout', 0.3),
                factor=self.args.get('factor', 5),
                pred_len=self.pred_len,
            )
        elif self.model_type == 'autoformer':
            kernel_sizes = self.args.get('kernel_sizes', [25])
            self.model = AutoformerDensityPredictor(
                input_size=input_size,
                future_cond_size=future_cond_size,
                d_model=self.args.get('d_model', 64),
                n_heads=self.args.get('n_heads', 4),
                e_layers=self.args.get('e_layers', 2),
                d_ff=self.args.get('d_ff', 512),
                dropout=self.args.get('dropout', 0.3),
                kernel_sizes=kernel_sizes,
                pred_len=self.pred_len,
            )
        elif self.model_type == 'lstm':
            self.model = LSTMMultiStep(
                input_size,
                future_cond_size,
                self.args.get('lstm_hidden', 128),
                self.args.get('lstm_layers', 2),
                self.args.get('dropout', 0.3),
                self.pred_len,
            )
        elif self.model_type == 'transformer':
            self.model = TransformerMultiStep(
                input_size,
                future_cond_size,
                self.args.get('d_model', 64),
                self.args.get('e_layers', 2),
                self.args.get('n_heads', 4),
                self.args.get('dropout', 0.3),
                self.pred_len,
            )
        elif self.model_type == 'tft':
            self.model = TemporalFusionTransformer(
                input_size=input_size,
                future_cond_size=future_cond_size,
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                d_model=self.args.get('d_model', 64),
                n_heads=self.args.get('n_heads', 4),
                num_layers=self.args.get('e_layers', 2),
                dropout=self.args.get('dropout', 0.3),
                lstm_hidden=self.args.get('lstm_hidden', 128),
                lstm_layers=self.args.get('lstm_layers', 2),
                use_variable_selection=self.args.get('use_variable_selection', True),
            )
        elif self.model_type == 'patchtst':
            self.model = PatchTST(
                input_size=input_size,
                future_cond_size=future_cond_size,
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                patch_len=self.args.get('patch_len', 16),
                stride=self.args.get('patch_stride', 8),
                d_model=self.args.get('d_model', 64),
                n_heads=self.args.get('n_heads', 4),
                num_layers=self.args.get('e_layers', 2),
                dropout=self.args.get('dropout', 0.3),
            )
        else:
            raise ValueError(f'不支持的模型类型: {self.model_type}')

        # 加载权重并进入推理模式
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.to(self.device)
        self.model.eval()

        print(f'[加载] 模型类型: {self.model_type}, seq_len={self.seq_len}, pred_len={self.pred_len}')
        print(f'[加载] 训练类型: {self.train_type}, 特征列数: {input_size}, 未来条件列数: {future_cond_size}')

    def _normalize_feature_matrix(self, data, columns):
        """
        对特征矩阵做与训练完全一致的归一化：
        1. 若该列存在于训练特征列中，则复用训练的 feature_min / feature_range
        2. 否则退回物理边界归一化
        3. 裁剪开启 clip 的物理边界
        """
        normed = data.copy().astype(np.float32)
        for i, col in enumerate(columns):
            if col in self.feature_columns:
                src_idx = self.feature_columns.index(col)
                # 按物理边界裁剪（与训练脚本 _clip_features 一致）
                if col in PHYSICAL_BOUNDS and PHYSICAL_BOUNDS[col] is not None and PHYSICAL_BOUNDS[col].get('clip', False):
                    bounds = PHYSICAL_BOUNDS[col]
                    normed[:, i] = np.clip(normed[:, i], bounds['min'], bounds['max'])
                # 使用训练时的 Min-Max 参数归一化
                normed[:, i] = (normed[:, i] - self.feature_min[src_idx]) / self.feature_range[src_idx]
            elif col in PHYSICAL_BOUNDS and PHYSICAL_BOUNDS[col] is not None:
                bounds = PHYSICAL_BOUNDS[col]
                if bounds.get('clip', False):
                    normed[:, i] = np.clip(normed[:, i], bounds['min'], bounds['max'])
                normed[:, i] = (normed[:, i] - bounds['min']) / max(bounds['max'] - bounds['min'], 1e-8)
        return normed

    def denormalize_prediction(self, pred):
        """将归一化预测值反变换回物理密度 (kg/m³)"""
        log_pred = (pred + 1.0) * 0.5 * self.target_log_range + self.target_log_min
        return np.power(10, log_pred)

    def predict_from_dataframe(self, df):
        """
        对整段轨迹做滑动多步预报。

        对每个有效起报点（需要足够历史 + 未来窗口），
        构造 (seq_len 历史特征, pred_len 未来显式条件) 输入，
        输出 pred_len 个物理密度预测值。

        Returns:
            predictions: (n_samples, pred_len) 密度预测数组
            start_indices: 每个样本在 df 中的未来窗口起始行号
        """
        n_windows = max(0, len(df) - self.seq_len - self.pred_len + 1)
        # 未来窗口起始行号（与原逐窗口逻辑一致：idx + seq_len）
        start_indices = np.arange(self.seq_len, self.seq_len + n_windows, dtype=np.int64)
        if n_windows == 0:
            return np.empty((0, self.pred_len), dtype=np.float64), start_indices

        # 一次性提取整段特征并归一化（归一化为全局仿射变换，先整体后切窗结果等价）
        feat_mat = self._normalize_feature_matrix(
            df[self.feature_columns].values.astype(np.float32), self.feature_columns)
        fut_mat = self._normalize_feature_matrix(
            df[self.future_condition_columns].values.astype(np.float32), self.future_condition_columns)

        # log_exp_residual 模式：网络输出是相对 NRLMSISE 的残差，反归一化前需加回基准
        use_residual = (self.train_type == 'log_exp_residual')
        nrlmsise_mat = None
        if use_residual:
            if NRLMSISE_COLUMN not in df.columns:
                raise ValueError(f'train_type={self.train_type} 需要 NRLMSISE 基准列，但数据中不存在 {NRLMSISE_COLUMN}')
            nrlmsise_mat = df[NRLMSISE_COLUMN].values.astype(np.float32)

        # 窗口内相对索引，用于分块取数
        seq_idx = np.arange(self.seq_len)
        fut_off = np.arange(self.pred_len)

        all_preds = np.empty((n_windows, self.pred_len), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, n_windows, INFER_BATCH_SIZE):
                e = min(s + INFER_BATCH_SIZE, n_windows)
                b = np.arange(s, e)  # 当前块内的窗口起始行号

                # (chunk, seq_len, F) 历史窗口 / (chunk, pred_len, C) 未来条件
                feat_batch = torch.from_numpy(
                    np.ascontiguousarray(feat_mat[b[:, None] + seq_idx[None, :]])).to(self.device)
                fut_batch = torch.from_numpy(
                    np.ascontiguousarray(fut_mat[(b[:, None] + self.seq_len) + fut_off[None, :]])).to(self.device)

                output = self.model(feat_batch, fut_batch)
                if isinstance(output, tuple):
                    output = output[0]  # Autoformer 返回 (输出, 尺度权重)

                # 残差模式：加回 NRLMSISE 基准（与训练时 target_nn = baseline_norm + output 一致）
                if use_residual:
                    nrl_batch = nrlmsise_mat[(b[:, None] + self.seq_len) + fut_off[None, :]]
                    log_baseline = np.log10(nrl_batch.astype(np.float64) + 1e-20)
                    baseline_norm = (2.0 * (log_baseline - self.target_log_min) / self.target_log_range - 1.0).astype(np.float32)
                    output = output + torch.from_numpy(baseline_norm).to(self.device)

                all_preds[s:e] = output.cpu().numpy()

        # 反归一化到物理密度（(n_windows, pred_len)）
        predictions = self.denormalize_prediction(all_preds)
        return predictions, start_indices


def compute_persist_nrlmsise(df, start_indices, pred_len, versions=(0, 2.0), ap_column=AP_COLUMN_DEFAULT):
    """
    按"起报时刻持久化"方式计算 NRLMSISE 密度（无预报值基线）。

    对每个起报窗口取起报时刻（最后一个历史观测帧）的 F10.7 / F10.7a / Ap 实测值，
    在整个预报期内保持不变；目标时刻的位置（经纬度、高度）取实际值。
    这与 NN 只用历史信息的设定对齐——NRLMSISE 不再吃目标时刻的未来真值。

    Args:
        df: 推理输入数据表（含时间/位置/太阳地磁列）
        start_indices: 各起报窗口的目标行起始行号
        pred_len: 预测步数
        versions: pymsis 版本列表（0=NRLMSISE-00, 2.0=NRLMSIS 2.0）
        ap_column: Ap 指数列名

    Returns:
        dict: {version: 密度数组}，数组长度与 (窗口, lead) 展开且剔除越界行后的行数一致
    """
    issue_idx = start_indices - 1
    f107 = df['space_environment_technologies__f107_obs__'].to_numpy(np.float64)[issue_idx]
    f107a = df['space_environment_technologies__f107_average__'].to_numpy(np.float64)[issue_idx]
    ap = df[ap_column].to_numpy(np.float64)[issue_idx]
    # 缺值处理（与数据处理脚本一致）：f107 缺省 150、f107a 缺省用 f107、ap 缺省 15
    f107 = np.where(np.isnan(f107), 150.0, f107)
    f107a = np.where(np.isnan(f107a), f107, f107a)
    ap = np.where(np.isnan(ap), 15.0, ap)

    # 目标行号（与展开表一致：(窗口, lead) C 顺序展平），并剔除越界行
    row_indices = (start_indices[:, None] + np.arange(int(pred_len))).ravel()
    valid_mask = row_indices < len(df)
    row_indices = row_indices[valid_mask]
    f107_rep = np.repeat(f107, pred_len)[valid_mask]
    f107a_rep = np.repeat(f107a, pred_len)[valid_mask]
    ap_rep = np.repeat(ap, pred_len)[valid_mask]

    dates = df[DATETIME_COLUMN].to_numpy()[row_indices]
    lons = df['tudelft_thermo__longitude__[deg]'].to_numpy(np.float64)[row_indices]
    lats = df['tudelft_thermo__latitude__[deg]'].to_numpy(np.float64)[row_indices]
    alts = df['tudelft_thermo__altitude__[m]'].to_numpy(np.float64)[row_indices] / 1000.0

    n_rows = len(row_indices)
    batch = 100000
    out = {}
    for ver in versions:
        dens = np.full(n_rows, np.nan)
        for s in range(0, n_rows, batch):
            e = min(s + batch, n_rows)
            ap_batch = np.repeat(ap_rep[s:e], 7).reshape(-1, 7)
            try:
                res = pymsis.calculate(dates[s:e], lons[s:e], lats[s:e], alts[s:e],
                                       f107_rep[s:e], f107a_rep[s:e], ap_batch, version=ver)
                dens[s:e] = res[:, 0]
            except Exception as ex:
                print(f'  [持久化基线] pymsis batch {s}:{e} (v={ver}) 失败: {ex}')
        out[ver] = dens
    return out


def main():
    parser = argparse.ArgumentParser(
        description='Forecasting 模型推理脚本（适配 train_forecasting_transformer.py 的 checkpoint）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ---------- 必选参数 ----------
    parser.add_argument('--model_path', type=str,default='models/finetune/patchtst_swarm_c_fold0_ds1_sl288_pl288_2014-01-01_to_2025-12-31_best.pth',
                        help='训练保存的模型 .pth 路径')
    parser.add_argument('--trajectory_file', type=str,default = 'data/merged_datasets/satellites_data_swarm_a_10xresample_omni.csv', 
                        help='卫星轨迹 CSV 路径')

    # ---------- 数据筛选参数 ----------
    parser.add_argument('--satellite', type=str, default='swarm_a', help='卫星名称，用于筛选卫星列')
    parser.add_argument('--start_time', type=str, default='2019-01-01 00:00:00', help='分析起始时间 (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end_time', type=str, default='2019-12-31 00:00:00', help='分析结束时间 (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--ap_column', type=str, default=AP_COLUMN_DEFAULT,
                        help='地磁 Ap 指数列名（用于分地磁相位分析）')

    # ---------- 运行参数 ----------
    parser.add_argument('--device', type=str, default='cuda', help='计算设备 (cpu 或 cuda)')

    # ---------- 可选输出参数（不指定则跳过对应功能） ----------
    parser.add_argument('--output', type=str, default='results',
                        help='输出父文件夹路径，结果将自动保存到其中带时间戳的子文件夹')
    parser.add_argument('--timestamp', action=argparse.BooleanOptionalAction, default=True,
                        help='是否在输出目录下创建带时间戳的子文件夹（默认开启；'
                             '--no-timestamp 时直接写入 --output 指定目录，不再生成子文件夹）')
    parser.add_argument('--plot', type=str, default='34',
                        help='绘图选项：none=不绘图；all=画图1~图4；数字组合如 "1,3" 或 "134" 只画对应编号的图（1=时序对比/2=逐步长MAPE/3=分相位MAPE/4=散点）')
    parser.add_argument('--plot_points', type=int, default=10000,
                        help='图4 散点图单系列最多绘制的点数（超限时按相同索引随机抽样，NN 与 NRLMSIS 点数始终一致；'
                             '设小值如 10 可逐点核对数量）')
    parser.add_argument('--summary', action=argparse.BooleanOptionalAction, default=True,
                        help='是否输出汇总 JSON（保存到输出目录的 summary.json；默认开启，--no-summary 关闭）')
    parser.add_argument('--persist_baseline', action='store_true',
                        help='额外计算无预报值（持久化）NRLMSISE 基线：起报时刻最后一次观测的 F10.7/Ap 在整个预报期内保持不变，'
                             '在输出 CSV 中增加 nrlmsise_density_persist（2.0）与 nrlmsise00_density_persist（00）两列，'
                             '并纳入汇总/报告对比')
    parser.add_argument('--report', action='store_true',
                        help='是否在控制台打印 NN vs NRLMSIS 三相位表')

    args = parser.parse_args()

    # 解析 --plot 参数（复用 infer_plotting 模块）：none=不绘图；all=图1~图4；数字组合只画对应编号的图
    plot_flags = parse_plot_flags(args.plot)

    # 输出目录：默认在 --output 下创建带时间戳的子文件夹；--no-timestamp 时直接写入 --output 指定目录
    out_dir = None
    if args.output:
        if args.timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            out_dir = os.path.join(args.output, timestamp)
        else:
            out_dir = args.output
        os.makedirs(out_dir, exist_ok=True)
        print(f'[输出] 本次结果目录: {out_dir}')

    # ---------- 1. 加载模型 ----------
    print(f'[加载] 模型: {args.model_path}')
    predictor = ForecastingPredictor(args.model_path, device=args.device)

    # ---------- 2. 读取并筛选数据 ----------
    df = pd.read_csv(args.trajectory_file)
    df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN])
    if SATELLITE_COLUMN in df.columns:
        df = df[df[SATELLITE_COLUMN] == args.satellite].copy()
    df = df.sort_values(DATETIME_COLUMN).reset_index(drop=True)

    # 预报目标时刻不超过 end_time
    if args.end_time:
        df = df[df[DATETIME_COLUMN] <= pd.to_datetime(args.end_time)].reset_index(drop=True)

    # start_time 是预报目标时刻的起点：历史特征应往前取 seq_len 行，而非在此处截断
    if args.start_time:
        start_ts = pd.to_datetime(args.start_time)
        # 第一个目标时刻 >= start_time 的行号
        first_target_idx = int(df[DATETIME_COLUMN].searchsorted(start_ts))
        # 往前保留 seq_len 行历史，保证第一个预报窗口就有完整历史输入
        hist_start_idx = max(0, first_target_idx - predictor.seq_len)
        df = df.iloc[hist_start_idx:].reset_index(drop=True)
    print(f'[数据] 有效行数: {len(df)}（含起报点前的 {predictor.seq_len} 行历史）')

    # ---------- 3. 逐窗口多步预报 ----------
    predictions, start_indices = predictor.predict_from_dataframe(df)

    # 判断数据中是否可用 NRLMSISE / 真实密度
    has_true = TARGET_COLUMN in df.columns
    has_nrlmsise = NRLMSISE_COLUMN in df.columns
    has_nrlmsise00 = NRLMSISE00_COLUMN in df.columns

    # ---------- 4. 展开为逐步长结果表（向量化构建，避免海量 Python dict 撑爆内存） ----------
    n_windows, pred_len = predictions.shape
    # 每个 (窗口, lead) 对应的目标行号 = 窗口起始行号 + lead（C 顺序展平，与 predictions 对齐）
    row_indices = (start_indices[:, None] + np.arange(pred_len)).ravel()
    lead_steps = np.tile(np.arange(1, pred_len + 1), n_windows)
    valid = row_indices < len(df)
    row_indices = row_indices[valid]
    lead_steps = lead_steps[valid]
    pred_flat = predictions.ravel()[valid]

    # 起报时刻 = 未来窗口前最后一帧（最新历史观测），每个窗口重复 pred_len 次
    issue_times = df[DATETIME_COLUMN].values[start_indices - 1]
    issue_rep = np.repeat(issue_times, pred_len)[valid]

    result = {
        'forecast_issue_time': pd.to_datetime(issue_rep),
        'target_time': pd.to_datetime(df[DATETIME_COLUMN].values[row_indices]),
        'lead_step': lead_steps,
        'prediction_density': pred_flat,
    }
    if has_true:
        result['true_density'] = df[TARGET_COLUMN].values[row_indices]
    if has_nrlmsise:
        result['nrlmsise_density'] = df[NRLMSISE_COLUMN].values[row_indices]
    if has_nrlmsise00:
        result['nrlmsise00_density'] = df[NRLMSISE00_COLUMN].values[row_indices]
    # 地磁相位（使用目标时刻的 Ap 值）
    if args.ap_column in df.columns:
        ap_vals = df[args.ap_column].values[row_indices]
        result['ap'] = ap_vals
        result['ap_phase'] = np.select(
            [ap_vals <= PHASE_QUIET_MAX, ap_vals <= PHASE_MINOR_MAX],
            ['quiet', 'minor'], default='major')

    df_result = pd.DataFrame(result)
    # 地磁相位转 category 以降低大结果表的内存占用
    if 'ap_phase' in df_result.columns:
        df_result['ap_phase'] = df_result['ap_phase'].astype('category')
    print(f'[结果] 展开逐步长样本数: {len(df_result)}')

    # 样本为空（数据行数不足以覆盖 seq_len+pred_len），提前返回避免后续汇总报错
    if len(df_result) == 0:
        print(f'[提示] 有效样本为 0：筛选后仅 {len(df)} 行，但至少需要 '
              f'seq_len({predictor.seq_len}) + pred_len({predictor.pred_len}) = '
              f'{predictor.seq_len + predictor.pred_len} 行。请放宽 --start_time/--end_time 时间范围。')
        return

    # ---------- 4.1 可选：无预报值（持久化）NRLMSISE 基线列 ----------
    # 用起报时刻最后一次观测的 F10.7/Ap 在整个预报期内保持不变，重新计算 NRLMSIS 2.0 与 00 密度，
    # 使 NRLMSISE 与 NN 一样不接触目标时刻的未来太阳/地磁真值
    has_persist = False
    if args.persist_baseline:
        if not PYMSIS_AVAILABLE:
            print('Warning: pymsis 不可用，跳过持久化 NRLMSISE 基线计算')
        elif args.ap_column not in df.columns:
            print(f'Warning: 未找到 Ap 列 {args.ap_column}，跳过持久化 NRLMSISE 基线计算')
        else:
            print('[持久化基线] 用起报时刻的 F10.7/Ap 计算 NRLMSIS 2.0 与 00 密度（预报期内保持不变）...')
            persist = compute_persist_nrlmsise(
                df, start_indices, predictor.pred_len, versions=(0, 2.0), ap_column=args.ap_column)
            df_result['nrlmsise00_density_persist'] = persist[0]
            df_result['nrlmsise_density_persist'] = persist[2.0]
            has_persist = True
            print(f'[持久化基线] 完成，已新增 nrlmsise_density_persist / nrlmsise00_density_persist 两列')

    # ---------- 5. 保存结果 CSV 到带时间戳的输出目录 ----------
    if out_dir:
        csv_path = os.path.join(out_dir, 'inference_results.csv')
        df_result.to_csv(csv_path, index=False)
        print(f'[输出] 结果已保存: {csv_path}')

    # 无真实密度时，无法做误差汇总 / 绘图 / 报告
    if not has_true:
        print('[提示] 数据中无真实密度列，跳过误差汇总、绘图与报告功能')
        return

    # ---------- 6. 可选：汇总（逐步长 + 分相位） ----------
    lead_metrics = None
    phase_metrics = None
    if args.summary or args.plot or args.report:
        lead_metrics = summarize_by_lead(df_result, has_nrlmsise, has_nrlmsise00, has_persist)
        phase_metrics = summarize_by_phase(df_result, has_nrlmsise, has_nrlmsise00, has_persist)

    if args.summary and out_dir:
        summary_path = os.path.join(out_dir, 'summary.json')
        summary = {
            'model_type': predictor.model_type,
            'seq_len': predictor.seq_len,
            'pred_len': predictor.pred_len,
            'train_type': predictor.train_type,
            'satellite': args.satellite,
            'start_time': str(df_result['target_time'].min()) if len(df_result) else None,
            'end_time': str(df_result['target_time'].max()) if len(df_result) else None,
            'lead_metrics': lead_metrics,
            'phase_metrics': phase_metrics,
            'overall': overall_metrics(df_result, has_nrlmsise, has_nrlmsise00, has_persist),
            'n_samples': int(len(df_result)),
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f'[输出] 汇总已保存: {summary_path}')

    # ---------- 7. 可选：自动绘图（--plot none/all/数字组合，按 plot_flags 绘制图1~图4） ----------
    if plot_flags and out_dir:
        plot_results(df_result, out_dir, predictor.model_type, has_nrlmsise, has_nrlmsise00,
                     plot_flags=plot_flags, max_scatter_points=args.plot_points)

    # ---------- 8. 可选：控制台打印三相位报告 ----------
    if args.report:
        overall = overall_metrics(df_result, has_nrlmsise, has_nrlmsise00, has_persist)
        print_report(phase_metrics, has_nrlmsise, predictor.model_type, has_nrlmsise00, has_persist, overall)


if __name__ == '__main__':
    main()
