"""
infer_plotting.py
=====================================================================
推理结果 CSV 分析绘图模块（可复用，也可独立运行）

功能：
  1. 读取推理输出 CSV（infer_forecasting_DL.py 生成的 inference_results.csv）
  2. 绘制四张图：
       图1：密度时序对比（True / NN / NRLMSIS 2.0 / NRLMSISE-00）
       图2：逐步长 MAPE 柱状对比
       图3：分地磁相位 MAPE 对比
       图4：观测密度 vs 预测密度散点（NN 与 NRLMSIS 2.0 两种点，1:1 图幅）
  3. --plot 支持 none / all / 数字组合（如 "1,3"、"34"），只画对应编号的图
  4. 汇总指标（逐步长 + 分相位 MAPE/RMSE）可输出 JSON / 控制台报告

作为独立脚本使用：
  python scripts/training/infer_plotting.py \
      --input results/xxx/inference_results.csv \
      --output results/xxx/plots \
      --plot all \
      --model_type patchtst

也可被 infer_forecasting_DL.py 以模块方式调用，复用同一套绘图逻辑。
=====================================================================
"""

import os
import sys
import argparse
import json
import re

import numpy as np
import pandas as pd

# 尝试导入 matplotlib；若不可用则仅跳过绘图功能
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端，适合脚本生成图片
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print('Warning: matplotlib 不可用，将跳过自动绘图功能')

# 结果表列名约定（与 infer_forecasting_DL.py 输出一致）
TIME_COL = 'target_time'
TRUE_COL = 'true_density'
PRED_COL = 'prediction_density'
NRL_COL = 'nrlmsise_density'                 # NRLMSIS 2.0（实测环境）
NRL00_COL = 'nrlmsise00_density'             # NRLMSISE-00（实测环境）
NRL_PERSIST_COL = 'nrlmsise_density_persist'     # NRLMSIS 2.0（持久化，无预报值）
NRL00_PERSIST_COL = 'nrlmsise00_density_persist' # NRLMSISE-00（持久化，无预报值）

# 地磁相位分档阈值（按 Ap 指数）：quiet ap<=30 ; minor 30<ap<=50 ; major ap>50
PHASE_QUIET_MAX = 30.0
PHASE_MINOR_MAX = 50.0

# 图4 散点图最多绘制的点数：数据量超限时随机抽样，避免千万级散点拖慢渲染
MAX_SCATTER_POINTS = 100000

# 模型图例显示名称（图4 右下角图例用）
MODEL_DISPLAY_NAMES = {
    'informer': 'Informer',
    'autoformer': 'Autoformer',
    'lstm': 'LSTM',
    'transformer': 'Transformer',
    'tft': 'TFT',
    'patchtst': 'PRC',
}


def _model_display_name(model_type):
    """返回模型图例显示名称（如 patchtst -> PatchTST）"""
    key = str(model_type).lower()
    return MODEL_DISPLAY_NAMES.get(key, model_type)


def _phase_of_ap(ap):
    """根据地磁 Ap 指数返回相位标签: 'quiet'(<=30) / 'minor'(30-50) / 'major'(>50)"""
    if ap <= PHASE_QUIET_MAX:
        return 'quiet'
    if ap <= PHASE_MINOR_MAX:
        return 'minor'
    return 'major'


def _compute_mape(pred, true):
    """计算 MAPE（%，带除零保护）"""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    mask = true > 1e-20
    if mask.sum() == 0:
        return float('nan')
    return float(np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100.0)


def _compute_rmse(pred, true):
    """计算 RMSE（物理空间，单位与密度一致）"""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def _compute_r(pred, true):
    """计算预测与真值的皮尔逊相关系数 R；样本不足 2 或方差为 0 时返回 NaN"""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if len(pred) < 2 or np.std(pred) == 0.0 or np.std(true) == 0.0:
        return float('nan')
    return float(np.corrcoef(pred, true)[0, 1])


def _compute_r2(pred, true):
    """计算决定系数 R2 = 1 - SS_res/SS_tot；真值方差为 0 时返回 NaN"""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    if ss_tot == 0.0:
        return float('nan')
    return 1.0 - ss_res / ss_tot


def overall_metrics(df_result, has_nrlmsise, has_nrlmsise00=False, has_persist=False):
    """
    计算整体（全部样本合并）总结指标：MAPE / RMSE / R / R2。

    Args:
        df_result: 推理结果表
        has_nrlmsise: 是否含 NRLMSIS 2.0 密度列
        has_nrlmsise00: 是否含 NRLMSISE-00 密度列
        has_persist: 是否含持久化（无预报值）NRLMSISE 基线列

    Returns:
        dict: 整体指标（含 n_samples 与各模型的 mape/rmse/r/r2）
    """
    true = df_result[TRUE_COL].values
    row = {
        'n_samples': int(len(df_result)),
        'nn_mape': _compute_mape(df_result[PRED_COL].values, true),
        'nn_rmse': _compute_rmse(df_result[PRED_COL].values, true),
        'nn_r': _compute_r(df_result[PRED_COL].values, true),
        'nn_r2': _compute_r2(df_result[PRED_COL].values, true),
    }
    for tag, col in (('nrlmsise', NRL_COL), ('nrlmsise00', NRL00_COL),
                     ('nrlmsise_persist', NRL_PERSIST_COL), ('nrlmsise00_persist', NRL00_PERSIST_COL)):
        # 按可用列跳过：2.0/00 按各自开关，persist 两列统一按 has_persist
        if (tag == 'nrlmsise' and not has_nrlmsise) or \
           (tag == 'nrlmsise00' and not has_nrlmsise00) or \
           (tag.endswith('persist') and not has_persist):
            continue
        row[f'{tag}_mape'] = _compute_mape(df_result[col].values, true)
        row[f'{tag}_rmse'] = _compute_rmse(df_result[col].values, true)
        row[f'{tag}_r'] = _compute_r(df_result[col].values, true)
        row[f'{tag}_r2'] = _compute_r2(df_result[col].values, true)
        # RMSE 相对该 MSIS 基线模型的降低百分比（正值 = NN 优于基线）
        row[f'nn_rmse_reduce_pct_{tag}'] = (row[f'{tag}_rmse'] - row['nn_rmse']) / row[f'{tag}_rmse'] * 100.0
    return row


def parse_plot_flags(plot_arg):
    """
    解析 --plot 参数：none=不绘图；all=图1~图4；数字组合（如 "1,3"、"34"）只画对应编号的图

    Returns:
        list[int]：需要绘制的图编号列表；空列表表示不绘图
    """
    flags = []
    if plot_arg:
        lower = str(plot_arg).lower().replace('，', ',')
        if lower not in ('none', '0'):
            if lower == 'all':
                flags = [1, 2, 3, 4]
            else:
                flags = sorted({int(c) for c in re.findall(r'\d', lower) if int(c) in (1, 2, 3, 4)})
    return flags


def detect_columns(df_result):
    """
    检测推理结果表中可用的对比列。

    Returns:
        (has_true, has_nrlmsise, has_nrlmsise00, has_persist)
    """
    has_true = TRUE_COL in df_result.columns
    has_nrlmsise = NRL_COL in df_result.columns
    has_nrlmsise00 = NRL00_COL in df_result.columns
    has_persist = NRL_PERSIST_COL in df_result.columns and NRL00_PERSIST_COL in df_result.columns
    return has_true, has_nrlmsise, has_nrlmsise00, has_persist


def summarize_by_lead(df_result, has_nrlmsise, has_nrlmsise00=False, has_persist=False):
    """
    按预测步长（lead step）汇总 MAPE/RMSE。

    Args:
        df_result: 推理结果表
        has_nrlmsise: 是否含 NRLMSIS 2.0 密度列
        has_nrlmsise00: 是否含 NRLMSISE-00 密度列
        has_persist: 是否含持久化（无预报值）NRLMSISE 基线列

    Returns:
        list[dict]：每个 lead step 一条指标
    """
    lead_metrics = []
    for lead, group in df_result.groupby('lead_step'):
        true = group[TRUE_COL].values
        row = {
            'lead_step': int(lead),
            'count': int(len(group)),
            'nn_mape': _compute_mape(group[PRED_COL].values, true),
            'nn_rmse': _compute_rmse(group[PRED_COL].values, true),
        }
        if has_nrlmsise:
            row['nrlmsise_mape'] = _compute_mape(group[NRL_COL].values, true)
            row['nrlmsise_rmse'] = _compute_rmse(group[NRL_COL].values, true)
        if has_nrlmsise00:
            row['nrlmsise00_mape'] = _compute_mape(group[NRL00_COL].values, true)
            row['nrlmsise00_rmse'] = _compute_rmse(group[NRL00_COL].values, true)
        if has_persist:
            row['nrlmsise_persist_mape'] = _compute_mape(group[NRL_PERSIST_COL].values, true)
            row['nrlmsise_persist_rmse'] = _compute_rmse(group[NRL_PERSIST_COL].values, true)
            row['nrlmsise00_persist_mape'] = _compute_mape(group[NRL00_PERSIST_COL].values, true)
            row['nrlmsise00_persist_rmse'] = _compute_rmse(group[NRL00_PERSIST_COL].values, true)
        lead_metrics.append(row)
    return lead_metrics


def summarize_by_phase(df_result, has_nrlmsise, has_nrlmsise00=False, has_persist=False):
    """
    分地磁相位汇总 MAPE：quiet ap<=30 / minor 30<ap<=50 / major ap>50

    Returns:
        dict: 以相位名为键，值为指标 dict；并附 'all' 总计
    """
    phase_order = ['quiet', 'minor', 'major']
    phase_metrics = {}
    for phase in phase_order:
        group = df_result[df_result['ap_phase'] == phase]
        if len(group) == 0:
            # 无样本相位：仍保留键，指标置 0（绘图时显示 0 柱，便于识别该相位无数据）
            row = {'count': 0, 'nn_mape': 0.0, 'nn_rmse': 0.0}
            if has_nrlmsise:
                row['nrlmsise_mape'] = 0.0
                row['nrlmsise_rmse'] = 0.0
            if has_nrlmsise00:
                row['nrlmsise00_mape'] = 0.0
                row['nrlmsise00_rmse'] = 0.0
            if has_persist:
                row['nrlmsise_persist_mape'] = 0.0
                row['nrlmsise_persist_rmse'] = 0.0
                row['nrlmsise00_persist_mape'] = 0.0
                row['nrlmsise00_persist_rmse'] = 0.0
            phase_metrics[phase] = row
            continue
        true = group[TRUE_COL].values
        row = {
            'count': int(len(group)),
            'nn_mape': _compute_mape(group[PRED_COL].values, true),
            'nn_rmse': _compute_rmse(group[PRED_COL].values, true),
        }
        if has_nrlmsise:
            row['nrlmsise_mape'] = _compute_mape(group[NRL_COL].values, true)
            row['nrlmsise_rmse'] = _compute_rmse(group[NRL_COL].values, true)
        if has_nrlmsise00:
            row['nrlmsise00_mape'] = _compute_mape(group[NRL00_COL].values, true)
            row['nrlmsise00_rmse'] = _compute_rmse(group[NRL00_COL].values, true)
        if has_persist:
            row['nrlmsise_persist_mape'] = _compute_mape(group[NRL_PERSIST_COL].values, true)
            row['nrlmsise_persist_rmse'] = _compute_rmse(group[NRL_PERSIST_COL].values, true)
            row['nrlmsise00_persist_mape'] = _compute_mape(group[NRL00_PERSIST_COL].values, true)
            row['nrlmsise00_persist_rmse'] = _compute_rmse(group[NRL00_PERSIST_COL].values, true)
        phase_metrics[phase] = row

    # 全部样本合计
    true = df_result[TRUE_COL].values
    total = {
        'count': int(len(df_result)),
        'nn_mape': _compute_mape(df_result[PRED_COL].values, true),
        'nn_rmse': _compute_rmse(df_result[PRED_COL].values, true),
    }
    if has_nrlmsise:
        total['nrlmsise_mape'] = _compute_mape(df_result[NRL_COL].values, true)
        total['nrlmsise_rmse'] = _compute_rmse(df_result[NRL_COL].values, true)
    if has_nrlmsise00:
        total['nrlmsise00_mape'] = _compute_mape(df_result[NRL00_COL].values, true)
        total['nrlmsise00_rmse'] = _compute_rmse(df_result[NRL00_COL].values, true)
    if has_persist:
        total['nrlmsise_persist_mape'] = _compute_mape(df_result[NRL_PERSIST_COL].values, true)
        total['nrlmsise_persist_rmse'] = _compute_rmse(df_result[NRL_PERSIST_COL].values, true)
        total['nrlmsise00_persist_mape'] = _compute_mape(df_result[NRL00_PERSIST_COL].values, true)
        total['nrlmsise00_persist_rmse'] = _compute_rmse(df_result[NRL00_PERSIST_COL].values, true)
    phase_metrics['all'] = total
    return phase_metrics


def plot_results(df_result, plot_dir, model_type, has_nrlmsise, has_nrlmsise00=False,
                 plot_flags=(1, 2, 3, 4), max_scatter_points=MAX_SCATTER_POINTS):
    """
    自动绘图（输出到文件夹，每张图独立保存，按 plot_flags 选择绘制）：
      图1：密度时序对比（True / NN / NRLMSIS 2.0 / NRLMSISE-00）
      图2：逐步长 MAPE 柱状对比
      图3：分地磁相位 MAPE 对比
      图4：观测密度 vs 预测密度散点（NN 与 NRLMSIS 2.0 两种点，x=y 参考线，1:1 图幅）

    Args:
        df_result: 推理结果表（含 target_time / true_density / prediction_density 等列）
        plot_dir: 输出目录
        model_type: 模型类型（图例显示，如 patchtst）
        has_nrlmsise: 是否含 NRLMSIS 2.0 密度列
        has_nrlmsise00: 是否含 NRLMSISE-00 密度列
        plot_flags: 需要绘制的图编号列表，如 [1, 3] 只画图1和图3
        max_scatter_points: 图4 单系列最多绘制的点数（超限时按相同索引随机抽样，
                            保证 NN 与 NRLMSIS 点数始终一致；设小值如 10 可逐点核对）
    """
    if not MATPLOTLIB_AVAILABLE:
        print('Warning: matplotlib 不可用，跳过绘图')
        return
    # 学术图表统一 Times New Roman，默认字号略调大
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    os.makedirs(plot_dir, exist_ok=True)

    df = df_result.sort_values(TIME_COL).copy()
    # 兼容独立脚本读取的 CSV（时间列为字符串）与推理流程（datetime64）
    if not np.issubdtype(df[TIME_COL].dtype, np.datetime64):
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    width = 0.25

    # ---------- 图1：密度时序对比（对数坐标） ----------
    if 1 in plot_flags:
        fig1, ax1 = plt.subplots(figsize=(14, 5))
        ax1.plot(df[TIME_COL], df[TRUE_COL], 'k-', lw=1.2, alpha=0.85, label='True')
        ax1.plot(df[TIME_COL], df[PRED_COL], 'b-', lw=1.2, alpha=0.85, label=_model_display_name(model_type))
        if has_nrlmsise:
            ax1.plot(df[TIME_COL], df[NRL_COL], 'r--', lw=1.2, alpha=0.85, label='NRLMSIS 2.0')
        if has_nrlmsise00:
            ax1.plot(df[TIME_COL], df[NRL00_COL], 'g--', lw=1.2, alpha=0.85, label='NRLMSISE-00')
        ax1.set_yscale('log')
        ax1.set_ylabel('Density (kg/m³)')
        ax1.set_title('Thermospheric Density Forecasting Comparison')
        ax1.legend(loc='upper right', fontsize=9, framealpha=0.6)
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        path1 = os.path.join(plot_dir, 'figure1_density_timeseries.png')
        fig1.savefig(path1, dpi=150, bbox_inches='tight')
        plt.close(fig1)
        print(f'[绘图] 图1 已保存: {path1}')

    # ---------- 图2：逐步长 MAPE 对比 ----------
    if 2 in plot_flags:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        lead_metrics = summarize_by_lead(df_result, has_nrlmsise, has_nrlmsise00)
        leads = [m['lead_step'] for m in lead_metrics]
        nn_mape = [m['nn_mape'] for m in lead_metrics]
        x = np.arange(len(leads))
        ax2.bar(x - width, nn_mape, width, label='NN', color='tab:blue')
        if has_nrlmsise:
            ax2.bar(x, [m['nrlmsise_mape'] for m in lead_metrics], width, label='NRLMSIS 2.0', color='tab:red')
        if has_nrlmsise00:
            ax2.bar(x + width, [m['nrlmsise00_mape'] for m in lead_metrics], width, label='NRLMSISE-00', color='tab:green')
        ax2.set_xticks(x, [str(l) for l in leads])
        ax2.set_xlabel('Lead step')
        ax2.set_ylabel('MAPE (%)')
        ax2.set_title('MAPE by Lead Step')
        ax2.legend()
        ax2.grid(True, axis='y', alpha=0.3)
        fig2.tight_layout()
        path2 = os.path.join(plot_dir, 'figure2_mape_by_lead.png')
        fig2.savefig(path2, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f'[绘图] 图2 已保存: {path2}')

    # ---------- 图3：分地磁活动水平(按 Ap) MAPE 对比 ----------
    if 3 in plot_flags:
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        phase_metrics = summarize_by_phase(df_result, has_nrlmsise, has_nrlmsise00)
        phase_order = [p for p in ['quiet', 'minor', 'major'] if p in phase_metrics] + (['all'] if 'all' in phase_metrics else [])
        # 分类沿用既有 quiet/minor/major（与历史图表一致，不重命名）
        nn_mape = [phase_metrics[p]['nn_mape'] for p in phase_order]
        x = np.arange(len(phase_order))
        # 学术配色：色盲友好的 muted 蓝/红/绿
        bar_colors = {'nn': '#2166AC', 'nrlmsise': '#D6604D', 'nrlmsise00': '#31A354'}
        nn_bar = ax3.bar(x - width, nn_mape, width, label='PRC', color=bar_colors['nn'],
                         edgecolor='white', linewidth=0.6, zorder=3)
        nrl_bar = nrl00_bar = None
        if has_nrlmsise:
            nrl_bar = ax3.bar(x, [phase_metrics[p]['nrlmsise_mape'] for p in phase_order],
                              width, label='NRLMSIS 2.0', color=bar_colors['nrlmsise'],
                              edgecolor='white', linewidth=0.6, zorder=3)
        if has_nrlmsise00:
            nrl00_bar = ax3.bar(x + width, [phase_metrics[p]['nrlmsise00_mape'] for p in phase_order],
                                width, label='NRLMSISE-00', color=bar_colors['nrlmsise00'],
                                edgecolor='white', linewidth=0.6, zorder=3)
        # x 轴刻度：分类名（不标注样本量）
        tick_labels = [p for p in phase_order]
        ax3.set_xticks(x, tick_labels, fontsize=11)
        # 柱顶标数值
        ymax = 0.0
        for i, p in enumerate(phase_order):
            ymax = max(ymax, phase_metrics[p]['nn_mape'],
                       phase_metrics[p].get('nrlmsise_mape', 0.0),
                       phase_metrics[p].get('nrlmsise00_mape', 0.0))
        offset = 0.02 * ymax
        for bars in (nn_bar, nrl_bar, nrl00_bar):
            if bars is None:
                continue
            for rect in bars:
                h = rect.get_height()
                ax3.text(rect.get_x() + rect.get_width() / 2, h + offset, f'{h:.1f}',
                         ha='center', va='bottom', fontsize=10)
        # 无样本相位在柱顶标注 n=0
        for i, p in enumerate(phase_order):
            if phase_metrics[p]['count'] == 0:
                ax3.text(i, 0, 'n=0', ha='center', va='bottom', fontsize=9, color='gray')
        ax3.set_ylim(0, ymax * 1.18)
        ax3.set_ylabel('MAPE (%)', fontsize=13)
        # 图例：左上角竖排，字号小一点
        ax3.legend(loc='upper left', frameon=False, fontsize=10)
        ax3.grid(True, axis='y', alpha=0.35, ls='--', zorder=0)
        for sp in ('top', 'right'):
            ax3.spines[sp].set_visible(False)
        fig3.tight_layout()
        path3 = os.path.join(plot_dir, 'figure3_mape_by_phase.png')
        fig3.savefig(path3, dpi=150, bbox_inches='tight')
        plt.close(fig3)
        print(f'[绘图] 图3 已保存: {path3}')

    # ---------- 图4：观测密度 vs 预测密度散点（NN 与 NRLMSIS 2.0 两种点，1:1图幅，无色柱） ----------
    if 4 in plot_flags:
        fig4, ax4 = plt.subplots(figsize=(7, 7))
        true = df[TRUE_COL].values
        pred = df[PRED_COL].values
        nrl = df[NRL_COL].values if has_nrlmsise else None
        valid = (true > 0) & (pred > 0) & np.isfinite(true) & np.isfinite(pred)
        if nrl is not None:
            valid &= (nrl > 0) & np.isfinite(nrl)
        if valid.sum() > 1:
            # R 基于全量计算
            r_nn = float(np.corrcoef(true[valid], pred[valid])[0, 1])
            r_nrl = float(np.corrcoef(true[valid], nrl[valid])[0, 1]) if nrl is not None else None
            # 数据量过大时随机抽样绘图（NN 与 NRLMSIS 共用同一组索引，保证两点一一对应同一条观测），
            # 避免千万级散点拖慢渲染
            t_v, p_v = true[valid], pred[valid]
            nrl_v = nrl[valid] if nrl is not None else None
            n_pts = len(t_v)
            if n_pts > max_scatter_points:
                idx = np.random.RandomState(0).choice(n_pts, max_scatter_points, replace=False)
                t_v, p_v = t_v[idx], p_v[idx]
                if nrl_v is not None:
                    nrl_v = nrl_v[idx]
            ax4.scatter(t_v, p_v, s=4, alpha=0.6, linewidths=0, rasterized=True,
                        color='tab:blue', label=_model_display_name(model_type))
            if nrl_v is not None:
                ax4.scatter(t_v, nrl_v, s=4, alpha=0.6, linewidths=0, rasterized=True,
                            color='tab:red', label='NRLMSIS 2.0')
            # x=y 参考线（不显示图例）
            lo = min(true[valid].min(), pred[valid].min())
            hi = max(true[valid].max(), pred[valid].max())
            ax4.plot([lo, hi], [lo, hi], 'k--', lw=1.2, alpha=0.7)
            ax4.set_xscale('log')
            ax4.set_yscale('log')
            ax4.set_aspect('equal')  # 图幅 1:1
            ax4.set_xlabel('Observed Density (kg m$^{-3}$)')
            ax4.set_ylabel('Predicted Density (kg m$^{-3}$)')
            ax4.set_title('Observed vs Predicted Density')
            # R 标注左上角（学术风格方框）
            r_text = f'$R_{{\\mathrm{{PRC}}}} = {r_nn:.4f}$'
            if r_nrl is not None:
                r_text += f'\n$R_{{\\mathrm{{NRLMSIS}}}} = {r_nrl:.4f}$'
            ax4.text(0.04, 0.96, r_text, transform=ax4.transAxes, fontsize=11,
                     verticalalignment='top',
                     bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='0.5', alpha=0.85))
            ax4.legend(loc='lower right', fontsize=9, framealpha=0.8)
            ax4.grid(True, which='both', alpha=0.3, lw=0.4)
            fig4.tight_layout()
            path4 = os.path.join(plot_dir, 'figure4_true_vs_pred_scatter.png')
            fig4.savefig(path4, dpi=150, bbox_inches='tight')
            plt.close(fig4)
            print(f'[绘图] 图4 已保存: {path4} (R_PRC={r_nn:.4f}, n = {int(valid.sum())})')
        else:
            plt.close(fig4)
            print('[绘图] 图4 跳过：有效正密度样本不足')


def print_report(phase_metrics, has_nrlmsise, model_type, has_nrlmsise00=False, has_persist=False, overall=None):
    """
    控制台打印推理报告：NN vs NRLMSIS 2.0 vs NRLMSISE-00 三相位 MAPE 表（可选）
    开启持久化基线时，额外列出无预报值（持久化）版本的 MAPE 列
    传入 overall（整体指标 dict）时，在表格后打印整体 MAPE/RMSE/R/R2 一行
    """
    phase_order = [p for p in ['quiet', 'minor', 'major'] if p in phase_metrics] + (['all'] if 'all' in phase_metrics else [])

    # 表头列：相位 / 样本数 / NN / [NRLMSIS 2.0] / [NRLMSISE-00] / [持久化2.0] / [持久化00] / NN RMSE
    header = f"{'Phase':<12}{'Count':>8}{'NN MAPE':>12}"
    if has_nrlmsise:
        header += f"{'NRLMSIS 2.0':>14}"
    if has_nrlmsise00:
        header += f"{'NRLMSISE-00':>14}"
    if has_persist:
        header += f"{'2.0持久化':>13}"
        header += f"{'00持久化':>13}"
    header += f"{'NN RMSE':>16}"

    sep = '-' * len(header)
    lines = [
        '',
        '=' * len(header),
        f'推理报告：{_model_display_name(model_type)} vs NRLMSIS 2.0 / NRLMSISE-00（按地磁相位）',
        '=' * len(header),
        header,
        sep,
    ]
    for phase in phase_order:
        m = phase_metrics[phase]
        line = f"{phase:<12}{m['count']:>8}{m['nn_mape']:>11.2f}%"
        if has_nrlmsise:
            line += f"{m['nrlmsise_mape']:>13.2f}%"
        if has_nrlmsise00:
            line += f"{m['nrlmsise00_mape']:>13.2f}%"
        if has_persist:
            line += f"{m['nrlmsise_persist_mape']:>12.2f}%"
            line += f"{m['nrlmsise00_persist_mape']:>12.2f}%"
        line += f"{m['nn_rmse']:>16.3e}"
        lines.append(line)
    lines.append(sep)
    # 整体（全部样本合并）指标：MAPE / RMSE / R / R2
    if overall:
        lines.append(f"整体指标（n={overall['n_samples']}）")
        for tag, disp in (('nn', 'NN'), ('nrlmsise', 'NRLMSIS 2.0'), ('nrlmsise00', 'NRLMSISE-00'),
                          ('nrlmsise_persist', '2.0持久化'), ('nrlmsise00_persist', '00持久化')):
            if f'{tag}_mape' not in overall:
                continue
            lines.append(
                f"  {disp:<13} MAPE={overall[f'{tag}_mape']:7.2f}%  RMSE={overall[f'{tag}_rmse']:.3e}  "
                f"R={overall[f'{tag}_r']:.4f}  R2={overall[f'{tag}_r2']:.4f}")
        lines.append(sep)
    print('\n'.join(lines))


def main():
    """独立运行入口：读取推理结果 CSV，绘图/汇总/报告"""
    parser = argparse.ArgumentParser(
        description='推理结果 CSV 分析绘图（读取 infer_forecasting_DL.py 输出的 CSV）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str, required=True, help='推理输出的 inference_results.csv 路径')
    parser.add_argument('--output', type=str, default=None, help='输出文件夹（默认与输入 CSV 同目录）')
    parser.add_argument('--plot', type=str, default='all',
                        help='绘图选项：none=不绘图；all=画图1~图4；数字组合如 "1,3" 或 "34" 只画对应编号的图（1=时序/2=逐步长MAPE/3=分相位MAPE/4=散点）')
    parser.add_argument('--model_type', type=str, default='NN',
                        help='模型名称（图例显示，如 patchtst / informer）')
    parser.add_argument('--summary', action='store_true', help='输出 summary.json（逐步长 + 分相位汇总）')
    parser.add_argument('--report', action='store_true', help='控制台打印分相位报告')
    args = parser.parse_args()

    df_result = pd.read_csv(args.input)
    df_result[TIME_COL] = pd.to_datetime(df_result[TIME_COL])
    has_true, has_nrlmsise, has_nrlmsise00, has_persist = detect_columns(df_result)
    print(f'[数据] 行数: {len(df_result)}, true={has_true}, nrlmsise2.0={has_nrlmsise}, '
          f'nrlmsise00={has_nrlmsise00}, 持久化基线={has_persist}')

    plot_dir = args.output if args.output else os.path.dirname(os.path.abspath(args.input))
    plot_flags = parse_plot_flags(args.plot)

    if plot_flags:
        plot_results(df_result, plot_dir, args.model_type, has_nrlmsise, has_nrlmsise00,
                     plot_flags=plot_flags)

    if not has_true:
        print('[提示] CSV 中无 true_density 列，跳过汇总与报告')
        return

    lead_metrics = None
    phase_metrics = None
    if args.summary or args.report or plot_flags:
        lead_metrics = summarize_by_lead(df_result, has_nrlmsise, has_nrlmsise00, has_persist)
        phase_metrics = summarize_by_phase(df_result, has_nrlmsise, has_nrlmsise00, has_persist)

    if args.summary:
        summary = {
            'model_type': args.model_type,
            'start_time': str(df_result[TIME_COL].min()),
            'end_time': str(df_result[TIME_COL].max()),
            'lead_metrics': lead_metrics,
            'phase_metrics': phase_metrics,
            'overall': overall_metrics(df_result, has_nrlmsise, has_nrlmsise00, has_persist),
            'n_samples': int(len(df_result)),
        }
        summary_path = os.path.join(plot_dir, 'summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f'[输出] 汇总已保存: {summary_path}')

    if args.report:
        overall = overall_metrics(df_result, has_nrlmsise, has_nrlmsise00, has_persist)
        print_report(phase_metrics, has_nrlmsise, args.model_type, has_nrlmsise00, has_persist, overall)


if __name__ == '__main__':
    main()
