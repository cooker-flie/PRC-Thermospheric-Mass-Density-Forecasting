"""
PRC 热层密度训练入口，保留论文对比模型实现
支持多步预测和数据降采样
"""

import os
import sys
import time
import argparse
import math
import pprint

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from tqdm import tqdm
import wandb
import random
import json
from datetime import datetime

# Windows 控制台默认 GBK 编码，print 含 ✓/✗ 等 Unicode 字符时会抛 UnicodeEncodeError。
# 将 stdout/stderr 的编码错误策略改为 replace（非法字符输出为 ?），避免训练中断。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors='replace')
    except (AttributeError, ValueError):
        pass

# 尝试导入matplotlib，如果失败则给出警告
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, training curves will not be saved")

# 设置固定随机种子，确保数据划分可重复
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
from pyfiglet import Figlet
from termcolor import colored

# 项目根目录（基于脚本位置解析，任意目录运行均可）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))

# 每次训练追加一条记录到该文件（wandb 目录下）
TRAINING_RUNS_LOG = os.path.join(_PROJECT_ROOT, 'wandb', 'training_runs.log')

# 本次进程的训练开始时间（在 train() 中设置，用于记录耗时）
_RUN_START_TIME = None


# ==================== 配置参数 ====================
# 物理边界归一化参数 - 基于物理意义和历史极端事件
# 避免分布偏移问题，训练/推理使用统一边界
PHYSICAL_BOUNDS = {
    # 太阳活动参数 - F10.7 范围覆盖历史太阳周期（2000-2024）
    'space_environment_technologies__f107_obs__': {'min': 50, 'max': 350, 'clip': True},
    'space_environment_technologies__f107_average__': {'min': 50, 'max': 350, 'clip': True},
    # 太阳活动代理指数
    'space_environment_technologies__s107_obs__': {'min': 50, 'max': 350, 'clip': True},
    'space_environment_technologies__s107_average__': {'min': 50, 'max': 350, 'clip': True},
    'space_environment_technologies__m107_obs__': {'min': 50, 'max': 350, 'clip': True},
    'space_environment_technologies__m107_average__': {'min': 50, 'max': 350, 'clip': True},
    'space_environment_technologies__y107_obs__': {'min': 50, 'max': 350, 'clip': True},
    'space_environment_technologies__y107_average__': {'min': 50, 'max': 350, 'clip': True},
    # 地磁活动指数 - 覆盖强磁暴事件
    'celestrack__ap_average__': {'min': 0, 'max': 150, 'clip': True},
    # 轨道几何参数
    'tudelft_thermo__altitude__[m]': {'min': 150000, 'max': 700000, 'clip': True},
    'tudelft_thermo__latitude__[deg]': {'min': -90, 'max': 90, 'clip': True},
    'tudelft_thermo__longitude__[deg]': {'min': -180, 'max': 180, 'clip': True},
    # 时间参数 - 角度类特征已在[-1,1]，无需归一化
    'local_time_sin': None,  # 保留原值
    'local_time_cos': None,  # 保留原值
    # 年和日 - 使用物理边界
    'all__year__[y]': {'min': 2000, 'max': 2030, 'clip': False},
    'all__day_of_year__[d]': {'min': 1, 'max': 366, 'clip': False},
    'all__seconds_in_day__[s]': {'min': 0, 'max': 86400, 'clip': False},
    # JB08参数
    'JB08__d_st_dt__[K]': {'min': -100, 'max': 100, 'clip': True},
    # OMNI参数 - 太阳风和地磁活动（基于数据统计调整）
    # IMF参数
    'omni__mag_avg_field_vector__[nT]': {'min': 0, 'max': 50, 'clip': True},
    'omni__bx_gse__[nT]': {'min': -50, 'max': 50, 'clip': True},
    'omni__by_gse__[nT]': {'min': -50, 'max': 50, 'clip': True},
    'omni__bz_gse__[nT]': {'min': -50, 'max': 50, 'clip': True},
    'omni__by_gsm__[nT]': {'min': -50, 'max': 50, 'clip': True},
    'omni__bz_gsm__[nT]': {'min': -50, 'max': 50, 'clip': True},
    # 太阳风等离子体
    'omni__flow_speed__[km/s]': {'min': 200, 'max': 1100, 'clip': True},
    'omni__vx__[km/s]': {'min': -1000, 'max': 1000, 'clip': True},
    'omni__vy__[km/s]': {'min': -1000, 'max': 1000, 'clip': True},
    'omni__vz__[km/s]': {'min': -1000, 'max': 1000, 'clip': True},
    'omni__proton_density__[n/cm3]': {'min': 0, 'max': 100, 'clip': True},
    'omni__temperature__[K]': {'min': 1000, 'max': 500000, 'clip': True},
    # 导出参数
    'omni__flow_pressure__[nPa]': {'min': 0, 'max': 100, 'clip': True},
    'omni__electric_field__[mV/m]': {'min': 0, 'max': 50, 'clip': True},
    'omni__plasma_beta__': {'min': 0, 'max': 10, 'clip': True},
    'omni__alfven_mach_number__': {'min': 0, 'max': 20, 'clip': True},
    # 地磁指数
    'omni__ae__[nT]': {'min': 0, 'max': 4000, 'clip': True},
    'omni__al__[nT]': {'min': -2000, 'max': 0, 'clip': True},
    'omni__au__[nT]': {'min': 0, 'max': 1000, 'clip': True},
    'omni__sym_d__[nT]': {'min': -500, 'max': 100, 'clip': True},
    'omni__sym_h__[nT]': {'min': -500, 'max': 100, 'clip': True},
    'omni__asym_d__[nT]': {'min': -200, 'max': 200, 'clip': True},
    'omni__asym_h__[nT]': {'min': -200, 'max': 200, 'clip': True},
    # RMS SD参数（磁场波动）
    'omni__rms_sd_b__[nT]': {'min': 0, 'max': 10, 'clip': True},
    'omni__rms_sd_field_vector__[nT]': {'min': 0, 'max': 10, 'clip': True},
    # 已剔除: na_np_ratio(38.98%缺失), mach_number(13.44%缺失)
}

# ---------- OMNI列选择配置（便捷修改区域）----------
# 完整可用OMNI列（基于新融合数据，已剔除1-13号参数和航天器位置）:
OMNI_ALL_COLUMNS = [
    # IMF参数 (含RMS波动)
    'omni__mag_avg_field_vector__[nT]',
    'omni__bx_gse__[nT]', 'omni__by_gse__[nT]', 'omni__bz_gse__[nT]',
    'omni__by_gsm__[nT]', 'omni__bz_gsm__[nT]',
    'omni__rms_sd_b__[nT]', 'omni__rms_sd_field_vector__[nT]',
    # 太阳风等离子体
    'omni__flow_speed__[km/s]', 'omni__vx__[km/s]', 'omni__vy__[km/s]', 'omni__vz__[km/s]',
    'omni__proton_density__[n/cm3]', 'omni__temperature__[K]',
    # 导出参数
    'omni__flow_pressure__[nPa]', 'omni__electric_field__[mV/m]',
    'omni__plasma_beta__', 'omni__alfven_mach_number__',
    # 地磁指数
    'omni__ae__[nT]', 'omni__al__[nT]', 'omni__au__[nT]',
    'omni__sym_d__[nT]', 'omni__sym_h__[nT]',
    'omni__asym_d__[nT]', 'omni__asym_h__[nT]',
    # 已剔除: na_np_ratio(38.98%缺失), mach_number(13.44%缺失)
]

# ===== 使用全部25个OMNI特征 =====
OMNI_SELECTED_COLUMNS = OMNI_ALL_COLUMNS  # 全维度：18基础 + 25 OMNI = 43维
# -----------------------------------------------

FEATURE_COLUMNS = [
    'tudelft_thermo__altitude__[m]',
    'tudelft_thermo__longitude__[deg]',
    'tudelft_thermo__latitude__[deg]',
    'local_time_sin',
    'local_time_cos',
    'all__year__[y]',
    'all__day_of_year__[d]',
    'all__seconds_in_day__[s]',
    'celestrack__ap_average__',
    'space_environment_technologies__f107_obs__',
    'space_environment_technologies__f107_average__',
    'space_environment_technologies__s107_obs__',
    'space_environment_technologies__s107_average__',
    'space_environment_technologies__m107_obs__',
    'space_environment_technologies__m107_average__',
    'space_environment_technologies__y107_obs__',
    'space_environment_technologies__y107_average__',
    'JB08__d_st_dt__[K]',
    # 历史密度特征（关键！让模型学习时间自相关性）
    'tudelft_thermo__ground_truth_thermospheric_density__[kg/m**3]',
] + OMNI_SELECTED_COLUMNS  # 追加选中的OMNI列

TARGET_COLUMN = 'tudelft_thermo__ground_truth_thermospheric_density__[kg/m**3]'
NRLMSISE_COLUMN = 'NRLMSIS20__thermospheric_density__[kg/m**3]'      # 主基线（残差学习基准 + NRLMSISE 对比）：NRLMSIS 2.0
NRLMSISE00_COLUMN = 'NRLMSISE00__thermospheric_density__[kg/m**3]'   # 测试集额外对比：NRLMSISE-00
DATETIME_COLUMN = 'all__dates_datetime__'
SATELLITE_COLUMN = 'tudelft_thermo__satellite__'

FUTURE_CONDITION_COLUMNS = [
    'tudelft_thermo__altitude__[m]',
    'tudelft_thermo__longitude__[deg]',
    'tudelft_thermo__latitude__[deg]',
    'local_time_sin',
    'local_time_cos',
    'all__year__[y]',
    'all__day_of_year__[d]',
    'all__seconds_in_day__[s]',
]


# ==================== Informer 模型组件 ====================

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class ProbSparseAttention(nn.Module):
    """ProbSparse Attention (Informer核心) - 正确实现"""
    def __init__(self, d_model, n_heads, dropout=0.1, factor=5):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, attn_mask=None):
        batch_size = q.size(0)
        
        q = self.q_proj(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        L_K = k.size(2)
        L_Q = q.size(2)
        
        # ProbSparse: 只对部分query计算attention
        if L_Q > self.factor:
            sample_size = max(1, L_Q // self.factor)
            # 选择稀疏query（按重要性采样，这里简化用随机）
            sample_idx = torch.randint(0, L_Q, (sample_size,), device=q.device)
            q_sample = q[:, :, sample_idx, :]
            
            # 只计算稀疏query的attention: O(sample_size * L_K) = O(L_Q/factor * L_K)
            scores_sample = torch.matmul(q_sample, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            
            if attn_mask is not None:
                scores_sample = scores_sample.masked_fill(attn_mask == 0, -1e9)
                
            attn_sample = F.softmax(scores_sample, dim=-1)
            attn_sample = self.dropout(attn_sample)
            
            # 对稀疏query做attention
            out_sample = torch.matmul(attn_sample, v)  # (batch, heads, sample_size, d_k)
            
            # 其余query用平均值（uniform attention）
            # v的均值: (batch, heads, 1, d_k) 
            v_mean = v.mean(dim=2, keepdim=True)  # 对所有key取平均
            
            # 合并结果
            out = torch.zeros(batch_size, self.n_heads, L_Q, self.d_k, device=q.device)
            out[:, :, sample_idx, :] = out_sample
            
            # 非采样query用uniform attention（等价于v的加权平均）
            all_idx = torch.arange(L_Q, device=q.device)
            non_sample_mask = torch.ones(L_Q, dtype=torch.bool, device=q.device)
            non_sample_mask[sample_idx] = False
            non_sample_idx = all_idx[non_sample_mask]
            
            if len(non_sample_idx) > 0:
                # uniform attention * v = mean(v)
                out[:, :, non_sample_idx, :] = v_mean
        else:
            # 短序列直接用标准attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask == 0, -1e9)
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
            
        out = out.transpose(1, 2).contiguous().view(batch_size, L_Q, self.d_model)
        return self.out_proj(out)


class AttentionDistilling(nn.Module):
    """自注意力蒸馏层 - 通过1D卷积压缩序列长度"""
    def __init__(self, d_model, dropout=0.1, pool_size=2):
        super().__init__()
        self.pool_size = pool_size
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=pool_size, stride=pool_size)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (B, L, D)
        # 转置为 (B, D, L) 进行1D卷积
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, L//pool, D)
        return self.norm(x)


class InformerEncoderLayer(nn.Module):
    """Informer编码器层 - 带注意力蒸馏"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, factor=5, do_distilling=True, pool_size=2):
        super().__init__()
        self.attention = ProbSparseAttention(d_model, n_heads, dropout, factor)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 注意力蒸馏
        self.do_distilling = do_distilling
        if do_distilling:
            self.distilling = AttentionDistilling(d_model, dropout, pool_size)
        else:
            self.distilling = None
        
    def forward(self, x, attn_mask=None):
        attn_out = self.attention(x, x, x, attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        # 应用蒸馏压缩序列
        if self.distilling is not None:
            x = self.distilling(x)
        
        return x


class FutureQueryDecoder(nn.Module):
    def __init__(self, d_model, future_cond_size, pred_len, dropout=0.1, n_heads=4):
        super().__init__()
        self.pred_len = pred_len
        self.lead_time_embedding = nn.Embedding(pred_len, d_model)
        self.future_projection = nn.Sequential(
            nn.Linear(future_cond_size, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, memory, future_conditions):
        batch_size, pred_len, _ = future_conditions.shape
        lead_idx = torch.arange(pred_len, device=memory.device).unsqueeze(0).expand(batch_size, -1)
        queries = self.lead_time_embedding(lead_idx) + self.future_projection(future_conditions)
        attn_out, _ = self.cross_attention(queries, memory, memory)
        x = self.norm1(queries + attn_out)
        x = self.norm2(x + self.ffn(x))
        return self.output_head(x).squeeze(-1)


class InformerDensityPredictor(nn.Module):
    """Informer密度预测模型 - 简化实现(与训练模型匹配)"""
    def __init__(self, input_size, future_cond_size, d_model=128, n_heads=4, e_layers=2, d_layers=1,
                 d_ff=512, dropout=0.1, factor=5, pred_len=12):
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        
        # 输入投影
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Encoder: 使用ProbSparse Attention，不蒸馏
        self.encoder_layers = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff, dropout, factor, do_distilling=False)
            for _ in range(e_layers)
        ])
        
        self.decoder = FutureQueryDecoder(d_model, future_cond_size, pred_len, dropout, n_heads)
        
    def forward(self, x, future_conditions=None):
        # Encoder
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        
        for layer in self.encoder_layers:
            x = layer(x)
        
        return self.decoder(x, future_conditions)


# ==================== Autoformer 模型组件 ====================

class MultiScaleSeriesDecomposition(nn.Module):
    """多尺度时间序列分解: 使用多个移动平均核捕捉不同周期"""
    def __init__(self, kernel_sizes=[18, 72, 288]):
        super().__init__()
        # 支持多个尺度: 18(~90min轨道), 72(~6h半日), 288(~24h日变化)
        self.kernel_sizes = kernel_sizes
        self.moving_avgs = nn.ModuleList([
            nn.AvgPool1d(kernel_size=k, stride=1, padding=k//2)
            for k in kernel_sizes
        ])
        # 可学习的融合权重
        self.scale_weights = nn.Parameter(torch.ones(len(kernel_sizes)) / len(kernel_sizes))
        
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        返回: seasonal, trend (多尺度融合后的)
        """
        x_perm = x.permute(0, 2, 1)  # (batch, d_model, seq_len)
        target_len = x.size(1)  # 目标序列长度
        
        # 多尺度趋势提取
        trends = []
        for avg_pool in self.moving_avgs:
            trend = avg_pool(x_perm)  # (batch, d_model, seq_len')
            # 确保输出长度与输入一致
            if trend.size(2) != target_len:
                trend = F.interpolate(trend, size=target_len, mode='linear', align_corners=False)
            trend = trend.permute(0, 2, 1)  # (batch, seq_len, d_model)
            trends.append(trend)
        
        # 加权融合多尺度趋势
        weights = F.softmax(self.scale_weights, dim=0)
        trend = sum(w * t for w, t in zip(weights, trends))
        seasonal = x - trend
        
        return seasonal, trend, weights  # 返回权重用于监控


class AutoCorrelation(nn.Module):
    """Auto-Correlation Mechanism (Autoformer核心)"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v):
        batch_size = q.size(0)
        
        q = self.q_proj(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        L = q.size(2)
        
        q_fft = torch.fft.rfft(q, dim=2)
        k_fft = torch.fft.rfft(k, dim=2)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), dim=2)
        
        top_k = min(5, L // 2)
        _, top_indices = torch.topk(corr.abs().mean(dim=(0, 1, 3)), top_k)
        
        out = torch.zeros_like(q)
        for delay in top_indices:
            delay = delay.item()
            if delay > 0 and delay < L:
                rolled_v = torch.roll(v, shifts=delay, dims=2)
                out += rolled_v
        out = out / top_k
        
        out = out.transpose(1, 2).contiguous().view(batch_size, L, self.d_model)
        return self.out_proj(out)


class AutoformerEncoderLayer(nn.Module):
    """Autoformer编码器层 (多尺度分解)"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, kernel_sizes=[18, 72, 288]):
        super().__init__()
        self.decomp1 = MultiScaleSeriesDecomposition(kernel_sizes)
        self.decomp2 = MultiScaleSeriesDecomposition(kernel_sizes)
        self.attention = AutoCorrelation(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_out = self.attention(x, x, x)
        seasonal, trend, w1 = self.decomp1(x + self.dropout(attn_out))
        x = seasonal
        
        ffn_out = self.ffn(x)
        seasonal, trend_part, w2 = self.decomp2(x + self.dropout(ffn_out))
        
        x = seasonal + trend + trend_part
        return x, w1  # 返回权重用于监控


class AutoformerDensityPredictor(nn.Module):
    """Autoformer密度预测模型 (多尺度版本)"""
    def __init__(self, input_size, future_cond_size, d_model=128, n_heads=4, e_layers=2,
                 d_ff=512, dropout=0.1, kernel_sizes=[18, 72, 288], pred_len=12):
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.kernel_sizes = kernel_sizes
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        self.encoder_layers = nn.ModuleList([
            AutoformerEncoderLayer(d_model, n_heads, d_ff, dropout, kernel_sizes)
            for _ in range(e_layers)
        ])
        
        self.decomp = MultiScaleSeriesDecomposition(kernel_sizes)
        
        self.decoder = FutureQueryDecoder(d_model, future_cond_size, pred_len, dropout, n_heads)
        
    def forward(self, x, future_conditions=None):
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        
        # 收集各层的多尺度权重用于监控
        scale_weights = []
        for layer in self.encoder_layers:
            x, weights = layer(x)
            scale_weights.append(weights)
        
        # 最终多尺度分解
        seasonal, trend, final_weights = self.decomp(x)
        
        output = self.decoder(x, future_conditions)
        
        # 返回输出和尺度权重（用于wandb监控）
        return output, scale_weights, final_weights


# ==================== LSTM 模型 ====================
class LSTMDensityPredictor(nn.Module):
    """LSTM 热层密度预测模型"""
    
    def __init__(self, input_size, future_cond_size, hidden_size=128, num_layers=2, dropout=0.2, pred_len=1):
        super(LSTMDensityPredictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_len = pred_len
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.decoder = FutureQueryDecoder(hidden_size, future_cond_size, pred_len, dropout, n_heads=4)
    
    def forward(self, x, future_conditions=None):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        return self.decoder(lstm_out, future_conditions)


# 别名，用于兼容train()函数中的引用
LSTMMultiStep = LSTMDensityPredictor


# ==================== Transformer 模型 ====================
class TransformerDensityPredictor(nn.Module):
    """Transformer 热层密度预测模型"""
    
    def __init__(self, input_size, d_model=128, num_layers=2, nhead=4, 
                 dim_feedforward=512, dropout=0.2, seq_len=24, activation='relu',
                 layer_norm_eps=1e-5):
        super(TransformerDensityPredictor, self).__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.seq_len = seq_len
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=seq_len)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)
        
        # Transformer 编码
        transformer_out = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # 取最后一个时间步的输出
        last_out = transformer_out[:, -1, :]  # (batch, d_model)
        output = self.fc(last_out)  # (batch, 1)
        return output.squeeze(-1)  # (batch,)


class TransformerMultiStep(nn.Module):
    """Transformer 多步热层密度预测模型"""
    
    def __init__(self, input_size, future_cond_size, d_model=128, num_layers=2, nhead=4, 
                 dropout=0.2, pred_len=12):
        super(TransformerMultiStep, self).__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.pred_len = pred_len
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 输出层 - 多步预测
        self.decoder = FutureQueryDecoder(d_model, future_cond_size, pred_len, dropout, nhead)
    
    def forward(self, x, future_conditions=None):
        # x: (batch, seq_len, input_size)
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        transformer_out = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        return self.decoder(transformer_out, future_conditions)


# ==================== TFT / PatchTST 模型 ====================

class GatedResidualNetwork(nn.Module):
    """门控残差网络 (GRN)"""
    def __init__(self, input_size, hidden_size, output_size=None, dropout=0.1, context_size=None):
        super().__init__()
        output_size = output_size or input_size
        self.input_size = input_size
        self.output_size = output_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
        # 门控机制
        self.gate = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
        # 上下文（静态变量）
        self.context_fc = None
        if context_size is not None:
            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)
        
        # 残差连接
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else None
        self.layer_norm = nn.LayerNorm(output_size)
    
    def forward(self, x, context=None):
        residual = x if self.skip is None else self.skip(x)
        
        hidden = self.fc1(x)
        if context is not None and self.context_fc is not None:
            hidden = hidden + self.context_fc(context)
        hidden = self.elu(hidden)
        
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)
        gate = self.sigmoid(self.gate(hidden))
        
        return self.layer_norm(residual + gate * output)


class VariableSelectionNetwork(nn.Module):
    """变量选择网络 (VSN) - TFT 核心可解释性模块"""
    def __init__(self, num_inputs, input_size, hidden_size, dropout=0.1, context_size=None):
        super().__init__()
        self.num_inputs = num_inputs
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 每个变量的GRN（输出维度应为hidden_size）
        self.single_variable_grns = nn.ModuleList([
            GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout=dropout)
            for _ in range(num_inputs)
        ])
        
        # 变量选择权重计算
        self.flattened_grn = GatedResidualNetwork(
            num_inputs * input_size, hidden_size, num_inputs,
            dropout=dropout, context_size=context_size
        )
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, context=None, return_weights=False):
        """
        x: (batch, num_inputs, input_size) - 各变量输入
        context: (batch, context_size) - 静态上下文
        return: (batch, hidden_size), 可选返回选择权重
        """
        batch_size = x.shape[0]
        
        # 计算变量选择权重
        flattened = x.reshape(batch_size, self.num_inputs * self.input_size)  # (batch, num_inputs * input_size)
        selection_weights = self.flattened_grn(flattened, context)  # (batch, num_inputs)
        selection_weights = self.softmax(selection_weights)  # (batch, num_inputs)
        
        # 对每个变量应用GRN
        variable_outputs = []
        for i, grn in enumerate(self.single_variable_grns):
            var_out = grn(x[:, i, :])  # (batch, hidden_size)
            variable_outputs.append(var_out)
        variable_outputs = torch.stack(variable_outputs, dim=1)  # (batch, num_inputs, hidden_size)
        
        # 加权求和
        weights_expanded = selection_weights.unsqueeze(-1)  # (batch, num_inputs, 1)
        weighted = variable_outputs * weights_expanded  # (batch, num_inputs, hidden_size)
        output = weighted.sum(dim=1)  # (batch, hidden_size)
        
        if return_weights:
            return output, selection_weights
        return output


class InterpretableMultiHeadAttention(nn.Module):
    """可解释多头注意力（TFT 核心）：各头独立 Q/K 权重、共享 V 权重以增强可解释性。

    与标准 nn.MultiheadAttention 的区别在于 value 投影只做一次并共享到所有头，
    使注意力权重可直接归因到输入变量的贡献。
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, 'd_model 必须能被 n_heads 整除'
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        # 共享 V 投影：所有头复用同一组 value 权重（interpretable 的关键）
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """query: (B, Lq, d_model), key/value: (B, Lk, d_model) -> (B, Lq, d_model)"""
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        q = self.q_proj(query).view(B, Lq, self.n_heads, self.d_k).transpose(1, 2)   # (B, H, Lq, dk)
        k = self.k_proj(key).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)     # (B, H, Lk, dk)
        v = self.v_proj(value).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)   # (B, H, Lk, dk) 共享 W_V

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, Lq, Lk)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)  # (B, H, Lq, dk)
        context = context.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.out_proj(context)


class TemporalFusionTransformer(nn.Module):
    """标准 TFT（Temporal Fusion Transformer）热层密度预测模型（单点输出版）。

    结构：变量选择(VSN) -> LSTM 局部时序编码 -> 可解释多头注意力(未来 query 查历史)
          -> 门控残差(GRN) -> 单点输出。
    与标准 TFT 的差异：为兼容项目现有单点 MSE + NRLMSISE 对比的评估体系，
    保留单点输出而非分位数输出。
    """
    def __init__(self, input_size, future_cond_size, seq_len=288, pred_len=12,
                 d_model=64, n_heads=4, num_layers=2, dropout=0.3,
                 lstm_hidden=128, lstm_layers=2, num_static_features=0,
                 use_variable_selection=True):
        super().__init__()
        self.input_size = input_size
        self.future_cond_size = future_cond_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.use_variable_selection = use_variable_selection

        # 静态特征编码（可选，默认 num_static_features=0 不启用）
        self.static_encoder = None
        if num_static_features > 0:
            self.static_encoder = GatedResidualNetwork(num_static_features, d_model, d_model, dropout=dropout)

        # 变量选择网络（历史 past 输入与未来 known 输入，逐时间步向量化应用）
        if use_variable_selection:
            self.past_vsn = VariableSelectionNetwork(
                num_inputs=input_size, input_size=1, hidden_size=d_model, dropout=dropout)
            self.future_vsn = VariableSelectionNetwork(
                num_inputs=future_cond_size, input_size=1, hidden_size=d_model, dropout=dropout)
        else:
            self.input_projection = nn.Linear(input_size, d_model)
            self.future_projection = nn.Linear(future_cond_size, d_model)

        # LSTM 局部时序编码器
        self.lstm = nn.LSTM(d_model, lstm_hidden, lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0)
        self.lstm_proj = nn.Linear(lstm_hidden, d_model) if lstm_hidden != d_model else nn.Identity()

        # 未来 query：提前量嵌入 + 未来条件
        self.lead_time_embedding = nn.Embedding(pred_len, d_model)

        # 可解释多头注意力
        self.interpretable_attention = InterpretableMultiHeadAttention(d_model, n_heads, dropout)

        # 输出门控残差 + 单点头
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout)
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, x, future_conditions=None):
        """x: (B, seq_len, input_size), future_conditions: (B, pred_len, future_cond_size) -> (B, pred_len)"""
        B, L, C = x.shape

        # 1. 变量选择（历史 past 输入，逐时间步向量化）
        if self.use_variable_selection:
            past_emb = self.past_vsn(x.reshape(B * L, C, 1)).reshape(B, L, self.d_model)  # (B, L, d_model)
        else:
            past_emb = self.input_projection(x)  # (B, L, d_model)

        # 2. LSTM 局部时序编码
        lstm_out, _ = self.lstm(past_emb)  # (B, L, lstm_hidden)
        lstm_out = self.lstm_proj(lstm_out)  # (B, L, d_model)

        # 3. 未来 query：提前量嵌入 + 未来 known 条件
        if future_conditions is None:
            future_conditions = torch.zeros(B, self.pred_len, self.future_cond_size, device=x.device)
        if self.use_variable_selection:
            P = future_conditions.shape[2]
            future_emb = self.future_vsn(future_conditions.reshape(B * self.pred_len, P, 1)).reshape(B, self.pred_len, self.d_model)
        else:
            future_emb = self.future_projection(future_conditions)

        lead_idx = torch.arange(self.pred_len, device=x.device).unsqueeze(0).expand(B, -1)
        queries = self.lead_time_embedding(lead_idx) + future_emb  # (B, pred_len, d_model)

        # 4. 可解释多头注意力：query=未来，key/value=历史 LSTM 输出
        attn_out = self.interpretable_attention(queries, lstm_out, lstm_out)  # (B, pred_len, d_model)
        x = self.norm1(queries + self.dropout(attn_out))

        # 5. 门控残差 + 单点输出
        x = self.output_grn(x)  # (B, pred_len, d_model)
        return self.output_head(x).squeeze(-1)  # (B, pred_len)


class PatchTST(nn.Module):
    """PatchTST 热层密度预测模型（标准 channel-independence 版）。

    每个输入变量独立 patch 化，并共享同一 Transformer 编码器（channel-independence），
    各变量独立预测后再聚合为单点多步输出。不使用未来显式条件（贴合原论文）。
    """
    def __init__(self, input_size, future_cond_size, seq_len=288, pred_len=12,
                 patch_len=16, stride=8, d_model=64, n_heads=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_size = input_size
        self.future_cond_size = future_cond_size  # 预留，channel-independence 不使用
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        self.num_patches = (seq_len - patch_len) // stride + 1
        if self.num_patches <= 0:
            raise ValueError(f'patch_len={patch_len} 大于 seq_len={seq_len}，无法切分 patch')

        # 共享 patch 投影 + 位置编码（所有变量复用）
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.patch_pos_encoding = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        self.dropout = nn.Dropout(dropout)

        # 共享 Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 共享输出头：展平 patch 特征 -> pred_len 步预测
        self.head = nn.Linear(self.num_patches * d_model, pred_len)

        # 聚合层：把 C 个变量的独立预测映射为最终单点预测
        self.aggregate = nn.Linear(input_size * pred_len, pred_len)

    def forward(self, x, future_conditions=None):
        """x: (B, seq_len, input_size) -> (B, pred_len)"""
        B, L, C = x.shape

        # channel-independence：合并 batch 与变量维度，每个变量独立送入共享编码器
        x = x.permute(0, 2, 1)  # (B, C, L)
        x = x.reshape(B * C, L)  # (B*C, L)

        # patch 化：(B*C, num_patches, patch_len)
        patches = x.unfold(1, self.patch_len, self.stride)

        # 投影 + 位置编码
        emb = self.patch_proj(patches) + self.patch_pos_encoding  # (B*C, num_patches, d_model)
        emb = self.dropout(emb)

        # 共享 Transformer 编码
        enc = self.encoder(emb)  # (B*C, num_patches, d_model)
        enc = enc.reshape(B * C, self.num_patches * self.d_model)

        # 每个变量独立预测
        out = self.head(enc)  # (B*C, pred_len)
        out = out.reshape(B, C, self.pred_len)  # (B, C, pred_len)

        # 聚合各变量预测为最终单点输出
        out = out.reshape(B, C * self.pred_len)
        return self.aggregate(out)  # (B, pred_len)


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ==================== 支持多步预测和降采样的数据集 ====================

class SatelliteDatasetMultiStep(Dataset):
    """支持多步预测和降采样的卫星数据集"""
    
    def __init__(self, csv_path, satellite, start_date, end_date, 
                 seq_len=144, pred_len=12, downsample=10, normalize=True,
                 norm_stats=None):
        """
        Args:
            seq_len: 输入序列长度（降采样后）
            pred_len: 预测步数（降采样后）
            downsample: 降采样因子（每N个点取1个，默认10=5分钟分辨率）
            norm_stats: 可选，外部归一化参数 dict（feature_min/feature_max/feature_range/
                        target_log_min/target_log_max/target_log_range）。用于时间外推测试集
                        复用训练集归一化参数，保证反归一化一致；None 则基于当前数据计算。
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.downsample = downsample
        self.normalize = normalize
        
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        mask = (df[SATELLITE_COLUMN] == satellite) & \
               (df[DATETIME_COLUMN] >= start_dt) & \
               (df[DATETIME_COLUMN] <= end_dt)
        df = df[mask].copy().reset_index(drop=True)
        
        if len(df) == 0:
            raise ValueError(f"No data found for satellite={satellite}, date range={start_date} to {end_date}")
        
        # 动态检测实际可用的特征列（适配含OMNI或不含OMNI的数据文件）
        available_cols = set(df.columns)
        self.effective_feature_columns = [c for c in FEATURE_COLUMNS if c in available_cols]
        skipped_cols = [c for c in FEATURE_COLUMNS if c not in available_cols]
        
        if skipped_cols:
            print(f"Warning: {len(skipped_cols)} feature columns not in data: {skipped_cols}")
        print(f"Using {len(self.effective_feature_columns)} features: {self.effective_feature_columns}")
        
        # 降采样
        print(f"Downsample factor: {downsample}")
        if downsample > 1:
            df = df.iloc[::downsample].reset_index(drop=True)
            print(f"After downsampling: {len(df)} samples")

        # 实际时间分辨率：中位时间步长（分钟），替代原先硬编码的 30 秒/步假设
        _diffs = df[DATETIME_COLUMN].diff().dropna().dt.total_seconds()
        _pos = _diffs[_diffs > 0]
        self.step_minutes = float(_pos.median() / 60.0) if len(_pos) else float('nan')
        print(f"Time resolution: ~{self.step_minutes:.1f} minutes (median step)")

        # 计算地方时（Local Time）特征
        print("Computing local time features...")
        seconds_in_day = df['all__seconds_in_day__[s]'].values
        longitude = df['tudelft_thermo__longitude__[deg]'].values
        
        # 地方时 = UTC小时 + 经度/15（每15度经度对应1小时）
        local_time_hours = (seconds_in_day / 3600.0 + longitude / 15.0) % 24.0
        
        # 正弦编码（捕捉周期性）
        df['local_time_sin'] = np.sin(2 * np.pi * local_time_hours / 24.0).astype(np.float32)
        df['local_time_cos'] = np.cos(2 * np.pi * local_time_hours / 24.0).astype(np.float32)
        
        self.dates = df[DATETIME_COLUMN].values
        self.future_condition_columns = [c for c in FUTURE_CONDITION_COLUMNS if c in df.columns]
        # 特征/未来条件缺失值处理：时序指数类（如 ap）缺值用前向+后向填充，
        # 避免 nan 进入网络导致损失与指标变 NaN（如 2025 下半年 ap 数据缺口）
        df[self.effective_feature_columns] = df[self.effective_feature_columns].ffill().bfill()
        df[self.future_condition_columns] = df[self.future_condition_columns].ffill().bfill()
        self.features = df[self.effective_feature_columns].values.astype(np.float32)
        self.future_conditions = df[self.future_condition_columns].values.astype(np.float32)
        self.raw_targets = df[TARGET_COLUMN].values.astype(np.float32)
        self.targets = self.raw_targets.copy()
        self.nrlmsise = df[NRLMSISE_COLUMN].values.astype(np.float32)
        self.nrlmsise_baseline = self.nrlmsise.copy()
        # NRLMSISE-00：仅用于测试集额外对比（主基线已切换为 NRLMSIS 2.0）
        self.nrlmsise00 = (df[NRLMSISE00_COLUMN].values.astype(np.float32)
                           if NRLMSISE00_COLUMN in df.columns else self.nrlmsise.copy())
        
        if normalize:
            if norm_stats is not None:
                # 复用外部归一化参数（用于时间外推测试集，保证与训练集反归一化一致）
                self.feature_min = norm_stats['feature_min']
                self.feature_max = norm_stats['feature_max']
                self.feature_range = norm_stats['feature_range']
                self.target_log_min = norm_stats['target_log_min']
                self.target_log_max = norm_stats['target_log_max']
                self.target_log_range = norm_stats['target_log_range']
            else:
                self._compute_normalization()
            # 先裁剪，再归一化
            self.features = self._clip_features(self.features)
            self.features = self._normalize_features(self.features)
            self.future_conditions = self._clip_future_conditions(self.future_conditions)
            self.future_conditions = self._normalize_future_conditions(self.future_conditions)
            self.targets = self._normalize_targets(self.targets)
        
        print(f"Feature shape: {self.features.shape}, Target shape: {self.targets.shape}")
        print(f"Effective time span per sample: input={seq_len * self.step_minutes:.1f}min, "
              f"pred={pred_len * self.step_minutes:.1f}min")
    
    def _compute_normalization(self):
        """使用物理边界进行归一化，避免分布偏移问题"""
        # 初始化归一化参数数组
        self.feature_min = np.zeros(len(self.effective_feature_columns))
        self.feature_max = np.zeros(len(self.effective_feature_columns))
        
        # 为每个特征设置物理边界
        for i, col in enumerate(self.effective_feature_columns):
            if col in PHYSICAL_BOUNDS:
                bounds = PHYSICAL_BOUNDS[col]
                if bounds is None:
                    # 无需归一化的特征（如三角函数编码），边界设为[-1, 1]
                    self.feature_min[i] = -1.0
                    self.feature_max[i] = 1.0
                else:
                    self.feature_min[i] = bounds['min']
                    self.feature_max[i] = bounds['max']
            else:
                # 对于未定义边界的特征，使用当前数据的统计值
                self.feature_min[i] = np.min(self.features[:, i])
                self.feature_max[i] = np.max(self.features[:, i])
        
        # 避免除零
        self.feature_range = self.feature_max - self.feature_min
        self.feature_range[self.feature_range == 0] = 1.0
        
        # 目标值归一化（保持原有逻辑，因为目标是密度，不是空间环境参数）
        log_targets = np.log10(self.targets)
        self.target_log_min = np.min(log_targets)
        self.target_log_max = np.max(log_targets)
        self.target_log_range = self.target_log_max - self.target_log_min + 1e-8
    
    def _clip_features(self, features):
        """根据物理边界裁剪特征值"""
        clipped = features.copy()
        for i, col in enumerate(self.effective_feature_columns):
            if col in PHYSICAL_BOUNDS and PHYSICAL_BOUNDS[col] is not None:
                bounds = PHYSICAL_BOUNDS[col]
                if bounds.get('clip', False):
                    clipped[:, i] = np.clip(clipped[:, i], bounds['min'], bounds['max'])
        return clipped
    
    def _normalize_features(self, features):
        """使用物理边界进行 Min-Max 归一化到 [0, 1]"""
        return (features - self.feature_min) / self.feature_range
    
    def _normalize_targets(self, targets):
        log_targets = np.log10(targets)
        return 2.0 * (log_targets - self.target_log_min) / self.target_log_range - 1.0

    def _clip_future_conditions(self, features):
        clipped = features.copy()
        for i, col in enumerate(self.future_condition_columns):
            if col in PHYSICAL_BOUNDS and PHYSICAL_BOUNDS[col] is not None:
                bounds = PHYSICAL_BOUNDS[col]
                if bounds.get('clip', False):
                    clipped[:, i] = np.clip(clipped[:, i], bounds['min'], bounds['max'])
        return clipped

    def _normalize_future_conditions(self, features):
        normed = features.copy()
        for i, col in enumerate(self.future_condition_columns):
            if col in self.effective_feature_columns:
                src_idx = self.effective_feature_columns.index(col)
                normed[:, i] = (features[:, i] - self.feature_min[src_idx]) / self.feature_range[src_idx]
            elif col in PHYSICAL_BOUNDS and PHYSICAL_BOUNDS[col] is not None:
                bounds = PHYSICAL_BOUNDS[col]
                normed[:, i] = (features[:, i] - bounds['min']) / max(bounds['max'] - bounds['min'], 1e-8)
        return normed
    
    def denormalize_prediction(self, pred):
        if not self.normalize:
            return pred
        log_pred = (pred + 1.0) * 0.5 * self.target_log_range + self.target_log_min
        return np.power(10, log_pred)
    
    def __len__(self):
        return max(0, len(self.features) - self.seq_len - self.pred_len + 1)
    
    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_len]
        
        y_start = idx + self.seq_len
        y = self.targets[y_start:y_start + self.pred_len]
        future_cond = self.future_conditions[y_start:y_start + self.pred_len]
        
        y_raw = self.raw_targets[y_start:y_start + self.pred_len]
        nrlmsise = self.nrlmsise[y_start:y_start + self.pred_len]
        nrlmsise_baseline = self.nrlmsise_baseline[y_start:y_start + self.pred_len]
        nrlmsise00 = self.nrlmsise00[y_start:y_start + self.pred_len]

        return {
            'features': torch.FloatTensor(x),
            'future_conditions': torch.FloatTensor(future_cond),
            'target': torch.FloatTensor(y),
            'ground_truth': torch.FloatTensor(y_raw),
            'nrlmsise': torch.FloatTensor(nrlmsise),
            'nrlmsise_baseline': torch.FloatTensor(nrlmsise_baseline),
            'nrlmsise00': torch.FloatTensor(nrlmsise00),
            'start_date': str(self.dates[y_start]),
            'end_date': str(self.dates[y_start + self.pred_len - 1])
        }


def mean_absolute_percentage_error(y_pred, y_true):
    """计算 MAPE，添加除零保护"""
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    # 避免除以0，过滤掉真实值接近0的样本
    mask = y_true > 1e-20
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def pearson_r(y_pred, y_true):
    """计算预测值与观测值的皮尔逊相关系数 R（样本不足 2 或方差为 0 时返回 NaN）"""
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float('nan')
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def r2_score(y_pred, y_true):
    """计算决定系数 R2 = 1 - SS_res/SS_tot（观测值方差为 0 时返回 NaN）"""
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float('nan')
    return float(1 - ss_res / ss_tot)


def mse(y_pred, y_true):
    """计算 MSE"""
    return np.mean((y_true - y_pred) ** 2)
def train_epoch(model, dataloader, criterion, optimizer, device, 
                losses_per_minibatch, losses_total, epoch, wandb_active, 
                train_type, dataset):
    """训练一个epoch (支持多步预测)"""
    model.train()
    loss_total_nn = 0.
    loss_total_nrlmsise = 0.
    mape_total_nn = 0.
    mape_total_nrlmsise = 0.
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        features = batch['features'].to(device)  # (batch, seq_len, features)
        future_conditions = batch['future_conditions'].to(device)
        target = batch['target'].to(device)  # (batch, pred_len)
        ground_truth = batch['ground_truth'].detach().cpu().numpy()  # (batch, pred_len)
        nrlmsise = batch['nrlmsise'].detach().cpu().numpy()  # (batch, pred_len)
        nrlmsise_baseline = batch['nrlmsise_baseline'].detach().cpu().numpy()
        
        optimizer.zero_grad()
        output = model(features, future_conditions)
        # 处理多返回值情况 (Autoformer 返回 tuple)
        if isinstance(output, tuple):
            output = output[0]  # 取第一个返回值 (实际预测输出)
        
        # 多步预测处理
        if train_type == 'log_density':
            # 直接预测对数归一化密度，无tanh限制
            target_nn = output
        elif train_type == 'log_exp_residual':
            # 残差学习模式：网络输入无NRLMSISE，但损失用残差形式
            # 对NRLMSISE基准进行归一化（仅用于损失计算，不参与网络输入）
            log_baseline = np.log10(nrlmsise_baseline + 1e-20)
            baseline_norm = 2.0 * (log_baseline - dataset.target_log_min) / dataset.target_log_range - 1.0
            baseline_norm_tensor = torch.FloatTensor(baseline_norm).to(device)
            
            # 网络输出作为残差（无tanh限制，让网络自由学习修正量）
            # 最终预测 = NRLMSISE基准 + 网络残差
            target_nn = baseline_norm_tensor + output  # output无tanh限制
        else:
            target_nn = output
        
        loss_nn = criterion(target_nn, target)
        loss_nn.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 评估指标计算（物理空间RMSE）
        pred_denorm = dataset.denormalize_prediction(target_nn.detach().cpu().numpy())
        rmse_physical = np.sqrt(np.mean((pred_denorm - ground_truth) ** 2))
        
        # 计算NRLMSISE的归一化损失（多步平均）
        log_nrlmsise = np.log10(np.maximum(nrlmsise, 1e-20))
        nrlmsise_norm = 2.0 * (log_nrlmsise - dataset.target_log_min) / dataset.target_log_range - 1.0
        loss_nrlmsise = np.mean((nrlmsise_norm - target.detach().cpu().numpy()) ** 2)
        
        # 记录到wandb（仅当wandb_active时）
        if wandb_active:
            wandb.log({
                'nn_mse_train': loss_nn.item(),
                'nn_rmse_physical_train': rmse_physical,
                'nrlmsise_mse_train': loss_nrlmsise,
                'nn_mape_train': mean_absolute_percentage_error(pred_denorm, ground_truth),
                'nrlmsise_mape_train': mean_absolute_percentage_error(nrlmsise, ground_truth)
            }, step=epoch)
        
        losses_per_minibatch['nn_mse_train'].append(loss_nn.item())
        losses_per_minibatch['nn_rmse_physical_train'].append(rmse_physical)
        losses_per_minibatch['nrlmsise_mse_train'].append(loss_nrlmsise)
        losses_per_minibatch['nn_mape_train'].append(mean_absolute_percentage_error(pred_denorm, ground_truth))
        losses_per_minibatch['nrlmsise_mape_train'].append(mean_absolute_percentage_error(nrlmsise, ground_truth))
        
        loss_total_nn += losses_per_minibatch['nn_mse_train'][-1]
        loss_total_nrlmsise += losses_per_minibatch['nrlmsise_mse_train'][-1]
        mape_total_nn += losses_per_minibatch['nn_mape_train'][-1]
        mape_total_nrlmsise += losses_per_minibatch['nrlmsise_mape_train'][-1]
    
    n_batches = len(dataloader)
    avg_loss = loss_total_nn / n_batches
    avg_rmse_physical = np.mean(losses_per_minibatch['nn_rmse_physical_train']) if losses_per_minibatch['nn_rmse_physical_train'] else 0
    losses_total['nn_mse_train'].append(avg_loss)
    losses_total['nn_rmse_physical_train'].append(avg_rmse_physical)
    losses_total['nrlmsise_mse_train'].append(loss_total_nrlmsise / n_batches)
    losses_total['nn_mape_train'].append(mape_total_nn / n_batches)
    losses_total['nrlmsise_mape_train'].append(mape_total_nrlmsise / n_batches)
    
    return avg_loss, loss_total_nn, avg_rmse_physical


def append_run_log(args, fold_idx, run_dir, best_val_mape,
                   best_nrlmsise_mape, test_mape, test_nrlmsise_mape):
    """将一次训练记录追加到 wandb/training_runs.log（每次训练一行块）。

    记录内容：开始/结束时间与耗时、训练数据时段与类型、数据文件、
    wandb 日志路径（本地目录）、本地日志目录、关键结果。

    参数:
        args: 训练配置。
        fold_idx: 折号。
        run_dir: 本地日志目录（local_logs 下的运行目录）。
        best_val_mape / best_nrlmsise_mape: 验证集最佳 MAPE。
        test_mape / test_nrlmsise_mape: 测试集 MAPE（无测试集时为 None）。
    """
    now = datetime.now()
    os.makedirs(os.path.dirname(TRAINING_RUNS_LOG), exist_ok=True)

    wandb_dir = ''
    if args.wandb_active and wandb.run is not None:
        wandb_dir = getattr(wandb.run, 'dir', '') or ''

    duration_str = 'N/A'
    if _RUN_START_TIME is not None:
        duration_str = str(now - _RUN_START_TIME).split('.')[0]  # 去掉微秒
    start_str = _RUN_START_TIME.strftime('%Y-%m-%d %H:%M:%S') if _RUN_START_TIME is not None else 'N/A'

    lines = [
        '=' * 72,
        f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] {args.model_type}_{args.satellite}_fold{fold_idx}",
        f"  模型         : {args.model_type}",
        f"  卫星         : {args.satellite}",
        f"  训练时段     : {args.start_date} ~ {args.end_date}",
        f"  训练类型     : {args.train_type}",
        f"  数据文件     : {os.path.abspath(args.data_path)}",
        f"  序列配置     : seq_len={args.seq_len}, pred_len={args.pred_len}, downsample={args.downsample}, k_folds={args.k_folds}",
        f"  开始时间     : {start_str}",
        f"  结束时间     : {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"  训练耗时     : {duration_str}",
        f"  本地日志目录 : {run_dir}",
        f"  wandb 目录   : {wandb_dir or '（未启用 --wandb_active）'}",
        f"  最佳结果     : val MAPE={best_val_mape:.2f}%, NRLMSISE MAPE={best_nrlmsise_mape:.2f}%",
    ]
    if test_mape is not None:
        lines.append(f"  测试结果     : test MAPE={test_mape:.2f}%, test NRLMSISE MAPE={test_nrlmsise_mape:.2f}%")

    with open(TRAINING_RUNS_LOG, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"\n[训练日志] 已追加: {TRAINING_RUNS_LOG}")


def save_local_logs(losses_total, args, dataset, fold_idx, best_val_mape, 
                    best_nrlmsise_mape, test_mape, test_nrlmsise_mape,
                    best_lead_metrics=None, test_lead_metrics=None,
                    test_nrlmsise00_mape=None,
                    test_rmse_physical=None, test_nrlmsise_mse=None,
                    test_nrlmsise00_mse=None, test_nrlmsise00_rmse=None,
                    test_nrlmsise_rmse=None,
                    test_nn_r=None, test_nn_r2=None,
                    test_nrl_r=None, test_nrl_r2=None,
                    test_nrl00_r=None, test_nrl00_r2=None):
    """
    保存训练日志和曲线图到本地（替代wandb功能）
    保存位置：当前发布项目根目录下的 wandb/
    """
    # 创建日志目录
    log_dir = os.path.join(_PROJECT_ROOT, 'wandb', 'local_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成运行名称
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{args.model_type}_{args.satellite}_fold{fold_idx}_{timestamp}"
    run_dir = os.path.join(log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\n[本地日志] 保存到: {run_dir}")
    
    # 1. 保存训练指标为JSON
    metrics = {
        'run_name': run_name,
        'timestamp': timestamp,
        'config': {
            'model_type': args.model_type,
            'satellite': args.satellite,
            'seq_len': args.seq_len,
            'pred_len': args.pred_len,
            'downsample': args.downsample,
            'train_type': args.train_type,
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'fold_idx': fold_idx,
        },
        'results': {
            'best_val_mape': float(best_val_mape),
            'best_nrlmsise_mape': float(best_nrlmsise_mape),
            'test_mape': float(test_mape) if test_mape is not None else None,
            'test_rmse_physical': float(test_rmse_physical) if test_rmse_physical is not None else None,
            'test_nn_r': float(test_nn_r) if test_nn_r is not None else None,
            'test_nn_r2': float(test_nn_r2) if test_nn_r2 is not None else None,
            'test_nrlmsise_mse': float(test_nrlmsise_mse) if test_nrlmsise_mse is not None else None,
            'test_nrlmsise_rmse': float(test_nrlmsise_rmse) if test_nrlmsise_rmse is not None else None,
            'test_nrlmsise_mape': float(test_nrlmsise_mape) if test_nrlmsise_mape is not None else None,
            'test_nrl_r': float(test_nrl_r) if test_nrl_r is not None else None,
            'test_nrl_r2': float(test_nrl_r2) if test_nrl_r2 is not None else None,
            'test_nrlmsise00_mse': float(test_nrlmsise00_mse) if test_nrlmsise00_mse is not None else None,
            'test_nrlmsise00_rmse': float(test_nrlmsise00_rmse) if test_nrlmsise00_rmse is not None else None,
            'test_nrlmsise00_mape': float(test_nrlmsise00_mape) if test_nrlmsise00_mape is not None else None,
            'test_nrl00_r': float(test_nrl00_r) if test_nrl00_r is not None else None,
            'test_nrl00_r2': float(test_nrl00_r2) if test_nrl00_r2 is not None else None,
            'best_val_lead_metrics': best_lead_metrics,
            'test_lead_metrics': test_lead_metrics,
        },
        'training_history': {
            'nn_mse_train': [float(x) for x in losses_total.get('nn_mse_train', [])],
            'nn_mse_valid': [float(x) for x in losses_total.get('nn_mse_valid', [])],
            'nn_mape_train': [float(x) for x in losses_total.get('nn_mape_train', [])],
            'nn_mape_valid': [float(x) for x in losses_total.get('nn_mape_valid', [])],
            'nrlmsise_mape_train': [float(x) for x in losses_total.get('nrlmsise_mape_train', [])],
            'nrlmsise_mape_valid': [float(x) for x in losses_total.get('nrlmsise_mape_valid', [])],
        }
    }
    
    json_path = os.path.join(run_dir, 'metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 指标已保存: metrics.json")
    
    # 2. 保存训练指标为CSV（便于Excel分析）
    epochs = len(losses_total.get('nn_mse_train', []))
    if epochs > 0:
        df_metrics = pd.DataFrame({
            'epoch': list(range(1, epochs + 1)),
            'train_mse': losses_total.get('nn_mse_train', [np.nan] * epochs),
            'valid_mse': losses_total.get('nn_mse_valid', [np.nan] * epochs),
            'train_mape': losses_total.get('nn_mape_train', [np.nan] * epochs),
            'valid_mape': losses_total.get('nn_mape_valid', [np.nan] * epochs),
            'nrlmsise_mape_train': losses_total.get('nrlmsise_mape_train', [np.nan] * epochs),
            'nrlmsise_mape_valid': losses_total.get('nrlmsise_mape_valid', [np.nan] * epochs),
        })
        csv_path = os.path.join(run_dir, 'training_log.csv')
        df_metrics.to_csv(csv_path, index=False)
        print(f"  ✓ 训练日志已保存: training_log.csv")
    
    # 3. 绘制并保存训练曲线图（2x3布局，增加RMSE）
    if MATPLOTLIB_AVAILABLE and epochs > 0:
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            
            # (0,0) MSE曲线
            ax1 = axes[0, 0]
            ax1.plot(losses_total['nn_mse_train'], label='Train MSE', color='blue')
            ax1.plot(losses_total['nn_mse_valid'], label='Valid MSE', color='orange')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('MSE')
            ax1.set_title('Training & Validation MSE')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # (0,1) MAPE曲线
            ax2 = axes[0, 1]
            ax2.plot(losses_total['nn_mape_train'], label='Train MAPE', color='blue')
            ax2.plot(losses_total['nn_mape_valid'], label='Valid MAPE', color='orange')
            ax2.axhline(y=best_nrlmsise_mape, color='red', linestyle='--', label=f'NRLMSISE ({best_nrlmsise_mape:.1f}%)')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAPE (%)')
            ax2.set_title('MAPE Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # (0,2) NN RMSE (phys)曲线 - 新增
            ax3 = axes[0, 2]
            ax3.plot(losses_total['nn_rmse_physical_train'], label='Train RMSE', color='blue')
            ax3.plot(losses_total['nn_rmse_physical_valid'], label='Valid RMSE', color='orange')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('RMSE (kg/m³)')
            ax3.set_title('NN Physical RMSE')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            
            # (1,0) NRLMSISE MAPE对比
            ax4 = axes[1, 0]
            ax4.plot(losses_total['nrlmsise_mape_train'], label='NRLMSISE Train', color='green')
            ax4.plot(losses_total['nrlmsise_mape_valid'], label='NRLMSISE Valid', color='red')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('MAPE (%)')
            ax4.set_title('NRLMSISE Baseline MAPE')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # (1,1) NRLMSISE RMSE (phys)曲线 - 新增
            ax5 = axes[1, 1]
            ax5.plot(losses_total.get('nrlmsise_rmse_physical_train', []), label='NRLMSISE Train RMSE', color='green')
            ax5.plot(losses_total.get('nrlmsise_rmse_physical_valid', []), label='NRLMSISE Valid RMSE', color='red')
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('RMSE (kg/m³)')
            ax5.set_title('NRLMSISE Physical RMSE')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            ax5.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            
            # (1,2) 汇总文本
            ax6 = axes[1, 2]
            ax6.axis('off')
            
            # 格式化RMSE值，处理科学计数法显示
            val_rmse = min(losses_total.get('nn_rmse_physical_valid', [0])) if losses_total.get('nn_rmse_physical_valid') else 0
            nrl_val_rmse = min(losses_total.get('nrlmsise_rmse_physical_valid', [0])) if losses_total.get('nrlmsise_rmse_physical_valid') else 0
            
            summary_text = f"""Training Summary
================
Model: {args.model_type}
Satellite: {args.satellite}
Fold: {fold_idx}
Train Type: {args.train_type}

Best Results (Validation)
-------------------------
Val MAPE: {best_val_mape:.2f}%
Val RMSE: {val_rmse:.3e}
NRLMSISE MAPE: {best_nrlmsise_mape:.2f}%
NRLMSISE RMSE: {nrl_val_rmse:.3e}
"""
            if test_mape is not None:
                test_rmse = min(losses_total.get('nn_rmse_physical_test', [0])) if losses_total.get('nn_rmse_physical_test') else 0
                test_nrl_rmse = min(losses_total.get('nrlmsise_rmse_physical_test', [0])) if losses_total.get('nrlmsise_rmse_physical_test') else 0
                summary_text += f"\nTest Results\n------------\n"
                summary_text += f"Test MAPE: {test_mape:.2f}%\n"
                summary_text += f"Test RMSE: {test_rmse:.3e}\n"
                summary_text += f"NRLMSISE MAPE: {test_nrlmsise_mape:.2f}%\n"
                summary_text += f"NRLMSISE RMSE: {test_nrl_rmse:.3e}\n"
            
            ax6.text(0.05, 0.5, summary_text, fontsize=8, family='monospace',
                    verticalalignment='center', horizontalalignment='left')
            
            plt.suptitle(f'Training Curves - {run_name}', fontsize=12)
            plt.tight_layout()
            
            plot_path = os.path.join(run_dir, 'training_curves.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ 曲线图已保存: training_curves.png")
        except Exception as e:
            print(f"  ✗ 绘图失败: {e}")
    
    # 追加简洁训练记录到 wandb/training_runs.log
    append_run_log(args, fold_idx, run_dir, best_val_mape,
                   best_nrlmsise_mape, test_mape, test_nrlmsise_mape)

    print(f"[本地日志] 保存完成\n")


def train_single_fold(model, train_loader, val_loader, test_loader, criterion, device, 
                      args, dataset, fold_idx, model_kwargs):
    """训练单个fold（用于K折交叉验证）"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.scheduler_patience
    )
    
    losses_per_minibatch = {
        'nn_mse_train': [], 'nn_rmse_physical_train': [], 'nrlmsise_mse_train': [], 'nn_mape_train': [], 'nrlmsise_mape_train': [],
        'nn_mse_valid': [], 'nn_rmse_physical_valid': [], 'nrlmsise_mse_valid': [], 'nn_mape_valid': [], 'nrlmsise_mape_valid': []
    }
    losses_total = {
        'nn_mse_train': [], 'nn_rmse_physical_train': [], 'nrlmsise_mse_train': [], 'nn_mape_train': [], 'nrlmsise_mape_train': [],
        'nn_mse_valid': [], 'nn_rmse_physical_valid': [], 'nrlmsise_mse_valid': [], 'nn_mape_valid': [], 'nrlmsise_mape_valid': []
    }
    
    best_val_loss = float('inf')
    best_val_lead_metrics = None
    patience_counter = 0
    patience = args.patience
    
    print(f"\nStarting training for fold {fold_idx + 1}...")
    print(f"Early stopping patience: {patience} epochs")
    print(f"Scheduler patience: {args.scheduler_patience} epochs")
    print("="*70)
    
    for epoch in range(args.epochs):
        train_loss, _, _ = train_epoch(
            model, train_loader, criterion, optimizer, device,
            losses_per_minibatch, losses_total, epoch, args.wandb_active,
            args.train_type, dataset
        )
        
        val_loss, val_mape, nrlmsise_mape, _, _, val_rmse_physical, nrlmsise_val_rmse, val_lead_metrics, _, _, _, _, _, _, _, _, _ = validate(
            model, val_loader, criterion, device, dataset,
            losses_per_minibatch, losses_total, epoch, args.wandb_active,
            args.train_type, args.pred_len
        )
        
        scheduler.step(val_loss)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_lead_metrics = val_lead_metrics
            patience_counter = 0
            # 保存最佳模型
            date_str = f"{args.start_date}_to_{args.end_date}"
            # 确保保存目录存在
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(
                args.save_dir,
                f"{args.model_type}_{args.satellite}_fold{fold_idx}_ds{args.downsample}_sl{args.seq_len}_pl{args.pred_len}_{date_str}_best.pth"
            )
            torch.save({
                'model_state_dict': model.state_dict(),
                'feature_min': dataset.feature_min,
                'feature_range': dataset.feature_range,
                'target_log_min': dataset.target_log_min,
                'target_log_max': dataset.target_log_max,
                'target_log_range': dataset.target_log_range,
                'args': vars(args),
                'feature_columns': dataset.effective_feature_columns,  # 保存实际使用的特征列
                'fold_idx': fold_idx,
                'normalization_type': 'physical_bounds'  # 标记归一化类型
            }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        # 每轮都打印进度
        train_rmse = losses_total['nn_rmse_physical_train'][-1] if losses_total['nn_rmse_physical_train'] else 0
        train_mape = losses_total['nn_mape_train'][-1] if losses_total['nn_mape_train'] else 0
        train_nrlmsise_mape = losses_total['nrlmsise_mape_train'][-1] if losses_total['nrlmsise_mape_train'] else 0
        val_rmse = losses_total['nn_rmse_physical_valid'][-1] if losses_total['nn_rmse_physical_valid'] else 0
        val_mape = losses_total['nn_mape_valid'][-1] if losses_total['nn_mape_valid'] else 0
        nrlmsise_mape = losses_total['nrlmsise_mape_valid'][-1] if losses_total['nrlmsise_mape_valid'] else 0
        print('\r', end='', flush=True)  # 清除tqdm残留
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"  Train - MSE: {losses_total['nn_mse_train'][-1]:.6f}, RMSE(phys): {train_rmse:.2e}, MAPE: {train_mape:.2f}%, NRLMSISE MAPE: {train_nrlmsise_mape:.2f}%")
        print(f"  Val   - MSE: {losses_total['nn_mse_valid'][-1]:.6f}, RMSE(phys): {val_rmse:.2e}, MAPE: {val_mape:.2f}%, NRLMSISE MAPE: {nrlmsise_mape:.2f}%")
    
    # 加载最佳模型进行测试评估（如果有测试集）
    if test_loader is not None:
        print(f"\nLoading best model for testing...")
        checkpoint = torch.load(save_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 在测试集上评估
        test_loss, test_mape, test_nrlmsise_mape, test_nrlmsise_mse, _, test_rmse_physical, test_nrlmsise_rmse, test_lead_metrics, test_nrlmsise00_mse, test_nrlmsise00_mape, test_nrlmsise00_rmse, test_nn_r, test_nn_r2, test_nrl_r, test_nrl_r2, test_nrl00_r, test_nrl00_r2 = validate(
            model, test_loader, criterion, device, dataset,
            {'nn_mse_valid': [], 'nn_rmse_physical_valid': [], 'nrlmsise_mse_valid': [],
             'nn_mape_valid': [], 'nrlmsise_mape_valid': []},
            {'nn_mse_valid': [], 'nn_rmse_physical_valid': [], 'nrlmsise_mse_valid': [],
             'nn_mape_valid': [], 'nrlmsise_mape_valid': []},
            0, False,  # 不需要记录损失历史
            args.train_type, args.pred_len
        )

        # 打印测试结果（NN vs NRLMSIS 2.0 vs NRLMSISE-00）
        print(f"\n{'='*60}")
        print(f"Test Results (Best Model)")
        print(f"{'='*60}")
        print(f"Test MSE: {test_loss:.6f}")
        print(f"Test RMSE (phys): {test_rmse_physical:.2e}")
        print(f"Test MAPE: {test_mape:.2f}%")
        print(f"Test R: {test_nn_r:.4f}, R2: {test_nn_r2:.4f}")
        print(f"Test NRLMSIS 2.0 MSE: {test_nrlmsise_mse:.6f}")
        print(f"Test NRLMSIS 2.0 RMSE (phys): {test_nrlmsise_rmse:.2e}")
        print(f"Test NRLMSIS 2.0 MAPE: {test_nrlmsise_mape:.2f}%")
        print(f"Test NRLMSIS 2.0 R: {test_nrl_r:.4f}, R2: {test_nrl_r2:.4f}")
        print(f"Test NRLMSISE-00 MSE: {test_nrlmsise00_mse:.6f}")
        print(f"Test NRLMSISE-00 RMSE (phys): {test_nrlmsise00_rmse:.2e}")
        print(f"Test NRLMSISE-00 MAPE: {test_nrlmsise00_mape:.2f}%")
        print(f"Test NRLMSISE-00 R: {test_nrl00_r:.4f}, R2: {test_nrl00_r2:.4f}")
        print(f"{'='*60}")
    else:
        # K折交叉验证，没有单独的测试集
        test_mape = test_loss = test_rmse_physical = test_nrlmsise_mape = None
        test_nrlmsise00_mse = test_nrlmsise00_mape = test_nrlmsise00_rmse = None
        test_nn_r = test_nn_r2 = test_nrl_r = test_nrl_r2 = test_nrl00_r = test_nrl00_r2 = None
        test_lead_metrics = None
    
    # 返回该fold的最佳结果
    best_val_mape = min(losses_total['nn_mape_valid']) if losses_total['nn_mape_valid'] else 100.0
    best_nrlmsise_mape = losses_total['nrlmsise_mape_valid'][np.argmin(losses_total['nn_mape_valid'])] if losses_total['nn_mape_valid'] else 30.0
    # 找到最佳MAPE对应的物理RMSE
    best_val_rmse_physical = losses_total['nn_rmse_physical_valid'][np.argmin(losses_total['nn_mape_valid'])] if losses_total['nn_mape_valid'] else 0.0
    
    # 打印最终汇总（供batch脚本解析）
    print(f"\n{'='*60}")
    print(f"Training Complete - Final Summary")
    print(f"{'='*60}")
    print(f"Best Val MAPE: {best_val_mape:.2f}%")
    print(f"Best Val MSE: {best_val_loss:.6f}")
    print(f"Best Val RMSE(phys): {best_val_rmse_physical:.2e}")
    print(f"NRLMSISE MAPE: {best_nrlmsise_mape:.2f}%")
    # 无测试集（K折/微调模式）时 test_* 为 None，统一打印 N/A
    if test_loader is not None:
        print(f"Test MAPE: {test_mape:.2f}%")
        print(f"Test MSE: {test_loss:.6f}")
        print(f"Test RMSE(phys): {test_rmse_physical:.2e}")
        print(f"Test R: {test_nn_r:.4f}, R2: {test_nn_r2:.4f}")
    else:
        print("Test MAPE: N/A (no test set)")
        print("Test MSE: N/A (no test set)")
        print("Test RMSE(phys): N/A (no test set)")
        print("Test R: N/A (no test set)")
    print(f"{'='*60}")
    
    # 保存本地日志和训练曲线（替代wandb功能）
    save_local_logs(losses_total, args, dataset, fold_idx, best_val_mape, 
                    best_nrlmsise_mape, test_mape, test_nrlmsise_mape,
                    best_val_lead_metrics, test_lead_metrics,
                    test_nrlmsise00_mape,
                    test_rmse_physical, test_nrlmsise_mse, test_nrlmsise00_mse,
                    test_nrlmsise00_rmse, test_nrlmsise_rmse,
                    test_nn_r, test_nn_r2, test_nrl_r, test_nrl_r2, test_nrl00_r, test_nrl00_r2)
    
    return best_val_loss, {
        'best_val_mape': best_val_mape,
        'best_nrlmsise_mape': best_nrlmsise_mape,
        'best_val_loss': best_val_loss,
        'test_mape': test_mape,
        'test_loss': test_loss,
        'test_rmse_physical': test_rmse_physical,
        'test_nrlmsise_mape': test_nrlmsise_mape,
        'test_nrlmsise00_mape': test_nrlmsise00_mape,
        'test_nn_r': test_nn_r,
        'test_nn_r2': test_nn_r2,
        'test_nrl_r': test_nrl_r,
        'test_nrl_r2': test_nrl_r2,
        'test_nrl00_r': test_nrl00_r,
        'test_nrl00_r2': test_nrl00_r2,
        'best_val_lead_metrics': best_val_lead_metrics,
        'test_lead_metrics': test_lead_metrics
    }


def validate(model, dataloader, criterion, device, dataset,
             losses_per_minibatch, losses_total, epoch, wandb_active, 
             train_type, pred_len):
    """验证 (支持多步预测)"""
    model.eval()
    loss_total_nn = 0.
    loss_total_nrlmsise = 0.
    mape_total_nn = 0.
    mape_total_nrlmsise = 0.
    loss_total_nrlmsise00 = 0.
    mape_total_nrlmsise00 = 0.
    rmse_physical_total = 0.  # 神经网络物理空间RMSE累计
    rmse_physical_nrlmsise_total = 0.  # NRLMSISE物理空间RMSE累计
    rmse_physical_nrlmsise00_total = 0.  # NRLMSISE-00物理空间RMSE累计
    lead_abs_pct_sum_nn = np.zeros(pred_len, dtype=np.float64)
    lead_abs_pct_sum_nrl = np.zeros(pred_len, dtype=np.float64)
    lead_sq_sum_nn = np.zeros(pred_len, dtype=np.float64)
    lead_sq_sum_nrl = np.zeros(pred_len, dtype=np.float64)
    lead_count = np.zeros(pred_len, dtype=np.float64)
    # 全量样本收集（用于计算 R / R2）
    all_gt, all_pred, all_nrl, all_nrl00 = [], [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation", leave=False)):
            features = batch['features'].to(device)
            future_conditions = batch['future_conditions'].to(device)
            target = batch['target'].to(device)
            ground_truth = batch['ground_truth'].detach().cpu().numpy()
            nrlmsise = batch['nrlmsise'].detach().cpu().numpy()
            nrlmsise_baseline = batch['nrlmsise_baseline'].detach().cpu().numpy()
            nrlmsise00 = batch['nrlmsise00'].detach().cpu().numpy()
            
            output = model(features, future_conditions)
            # 处理多返回值情况
            if isinstance(output, tuple):
                output = output[0]
            
            if train_type == 'log_density':
                # 直接预测对数归一化密度，无tanh限制
                target_nn = output
            elif train_type == 'log_exp_residual':
                # 残差学习：网络预测残差，与NRLMSISE相加得到最终预测
                # 无tanh限制，让网络自由学习修正量（训练和验证一致）
                log_baseline = np.log10(nrlmsise_baseline + 1e-20)
                baseline_norm = 2.0 * (log_baseline - dataset.target_log_min) / dataset.target_log_range - 1.0
                baseline_norm_tensor = torch.FloatTensor(baseline_norm).to(device)
                # 网络输出直接作为残差，无tanh限制
                target_nn = baseline_norm_tensor + output
            else:
                target_nn = output
            
            loss_nn = criterion(target_nn, target)
            loss_total_nn += loss_nn.item()
            
            pred_denorm = dataset.denormalize_prediction(target_nn.cpu().numpy())
            
            # 收集全量样本用于计算 R / R2
            all_gt.append(ground_truth)
            all_pred.append(pred_denorm)
            all_nrl.append(nrlmsise)
            all_nrl00.append(nrlmsise00)
            
            # 计算物理空间RMSE
            rmse_physical = np.sqrt(np.mean((pred_denorm - ground_truth) ** 2))
            rmse_physical_total += rmse_physical
            
            log_nrlmsise = np.log10(np.maximum(nrlmsise, 1e-20))
            nrlmsise_norm = 2.0 * (log_nrlmsise - dataset.target_log_min) / dataset.target_log_range - 1.0
            loss_nrlmsise = np.mean((nrlmsise_norm - target.cpu().numpy()) ** 2)
            loss_total_nrlmsise += loss_nrlmsise
            
            mape_nn = mean_absolute_percentage_error(pred_denorm, ground_truth)
            mape_nrlmsise = mean_absolute_percentage_error(nrlmsise, ground_truth)
            mask = ground_truth > 1e-20
            lead_abs_pct_sum_nn += np.sum(np.where(mask, np.abs((ground_truth - pred_denorm) / ground_truth), 0.0), axis=0)
            lead_abs_pct_sum_nrl += np.sum(np.where(mask, np.abs((ground_truth - nrlmsise) / ground_truth), 0.0), axis=0)
            lead_sq_sum_nn += np.sum((pred_denorm - ground_truth) ** 2, axis=0)
            lead_sq_sum_nrl += np.sum((nrlmsise - ground_truth) ** 2, axis=0)
            lead_count += np.sum(mask, axis=0)
            
            # 计算NRLMSISE物理空间RMSE
            rmse_physical_nrlmsise = np.sqrt(np.mean((nrlmsise - ground_truth) ** 2))
            rmse_physical_nrlmsise_total += rmse_physical_nrlmsise
            
            mape_total_nn += mape_nn
            mape_total_nrlmsise += mape_nrlmsise

            # 计算 NRLMSISE-00 额外对比指标（仅测试集评价使用，但统一累计便于复用）
            log_nrlmsise00 = np.log10(np.maximum(nrlmsise00, 1e-20))
            nrlmsise00_norm = 2.0 * (log_nrlmsise00 - dataset.target_log_min) / dataset.target_log_range - 1.0
            loss_nrlmsise00 = np.mean((nrlmsise00_norm - target.cpu().numpy()) ** 2)
            loss_total_nrlmsise00 += loss_nrlmsise00
            mape_nrlmsise00 = mean_absolute_percentage_error(nrlmsise00, ground_truth)
            mape_total_nrlmsise00 += mape_nrlmsise00
            rmse_physical_nrlmsise00 = np.sqrt(np.mean((nrlmsise00 - ground_truth) ** 2))
            rmse_physical_nrlmsise00_total += rmse_physical_nrlmsise00
            
            if wandb_active:
                wandb.log({
                    'nn_mse_valid': loss_nn.item(),
                    'nn_rmse_physical_valid': rmse_physical,
                    'nrlmsise_mse_valid': loss_nrlmsise,
                    'nn_mape_valid': mape_nn,
                    'nrlmsise_mape_valid': mape_nrlmsise
                }, step=epoch)
            
            losses_per_minibatch['nn_mse_valid'].append(loss_nn.item())
            losses_per_minibatch['nn_rmse_physical_valid'].append(rmse_physical)
            losses_per_minibatch['nrlmsise_mse_valid'].append(loss_nrlmsise)
            losses_per_minibatch['nn_mape_valid'].append(mape_nn)
            losses_per_minibatch['nrlmsise_mape_valid'].append(mape_nrlmsise)
    
    n_batches = len(dataloader)
    avg_loss = loss_total_nn / n_batches
    avg_nrlmsise_loss = loss_total_nrlmsise / n_batches
    avg_mape = mape_total_nn / n_batches
    avg_nrlmsise_mape = mape_total_nrlmsise / n_batches
    avg_rmse_physical = rmse_physical_total / n_batches  # 平均神经网络物理空间RMSE
    avg_rmse_physical_nrlmsise = rmse_physical_nrlmsise_total / n_batches  # 平均NRLMSISE物理空间RMSE
    lead_count = np.maximum(lead_count, 1.0)
    lead_metrics = []
    for lead_idx in range(pred_len):
        lead_metrics.append({
            'lead_step': int(lead_idx + 1),
            'lead_minutes': float((lead_idx + 1) * dataset.step_minutes),
            'nn_mape': float(lead_abs_pct_sum_nn[lead_idx] / lead_count[lead_idx] * 100.0),
            'nrlmsise_mape': float(lead_abs_pct_sum_nrl[lead_idx] / lead_count[lead_idx] * 100.0),
            'nn_rmse': float(np.sqrt(lead_sq_sum_nn[lead_idx] / lead_count[lead_idx])),
            'nrlmsise_rmse': float(np.sqrt(lead_sq_sum_nrl[lead_idx] / lead_count[lead_idx]))
        })

    losses_total['nn_mse_valid'].append(avg_loss)
    losses_total['nn_rmse_physical_valid'].append(avg_rmse_physical)
    losses_total['nrlmsise_mse_valid'].append(avg_nrlmsise_loss)
    losses_total['nn_mape_valid'].append(avg_mape)
    losses_total['nrlmsise_mape_valid'].append(avg_nrlmsise_mape)

    avg_nrlmsise00_loss = loss_total_nrlmsise00 / n_batches
    avg_nrlmsise00_mape = mape_total_nrlmsise00 / n_batches
    avg_rmse_physical_nrlmsise00 = rmse_physical_nrlmsise00_total / n_batches

    # 拼接全量样本计算 R / R2（NN / NRLMSIS 2.0 / NRLMSISE-00 三套）
    gt_all = np.concatenate(all_gt).flatten() if all_gt else np.array([])
    pred_all = np.concatenate(all_pred).flatten() if all_pred else np.array([])
    nrl_all = np.concatenate(all_nrl).flatten() if all_nrl else np.array([])
    nrl00_all = np.concatenate(all_nrl00).flatten() if all_nrl00 else np.array([])
    nn_r = pearson_r(pred_all, gt_all)
    nn_r2 = r2_score(pred_all, gt_all)
    nrl_r = pearson_r(nrl_all, gt_all)
    nrl_r2 = r2_score(nrl_all, gt_all)
    nrl00_r = pearson_r(nrl00_all, gt_all)
    nrl00_r2 = r2_score(nrl00_all, gt_all)

    return (avg_loss, avg_mape, avg_nrlmsise_mape, avg_nrlmsise_loss, loss_total_nn,
            avg_rmse_physical, avg_rmse_physical_nrlmsise, lead_metrics,
            avg_nrlmsise00_loss, avg_nrlmsise00_mape, avg_rmse_physical_nrlmsise00,
            nn_r, nn_r2, nrl_r, nrl_r2, nrl00_r, nrl00_r2)


def build_holdout_test_loader(args, train_dataset, start_date, end_date):
    """构建时间外推测试集。

    测试集从 end_date 后 test_gap_days 天起，取训练时段总时长的 test_ratio 作为时长，
    复用训练集的归一化参数。若该时段数据不足，返回 None 以便回退到原末尾划分。

    Args:
        args: 解析后的训练参数（需含 data_path/satellite/test_ratio/test_gap_days/
              seq_len/pred_len/downsample/batch_size）
        train_dataset: 训练集 dataset，用于复用其归一化参数
        start_date: 训练时段起始（str）
        end_date: 训练时段结束（str）

    Returns:
        DataLoader 或 None（数据不足时）
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    test_span = (end_dt - start_dt) * args.test_ratio
    test_start = end_dt + pd.Timedelta(days=args.test_gap_days)
    test_end = test_start + test_span

    norm_stats = {
        'feature_min': train_dataset.feature_min,
        'feature_max': train_dataset.feature_max,
        'feature_range': train_dataset.feature_range,
        'target_log_min': train_dataset.target_log_min,
        'target_log_max': train_dataset.target_log_max,
        'target_log_range': train_dataset.target_log_range,
    }

    try:
        test_dataset = SatelliteDatasetMultiStep(
            csv_path=args.data_path,
            satellite=args.satellite,
            start_date=test_start.strftime('%Y-%m-%d %H:%M:%S'),
            end_date=test_end.strftime('%Y-%m-%d %H:%M:%S'),
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            downsample=args.downsample,
            normalize=True,
            norm_stats=norm_stats,
        )
    except ValueError:
        return None

    if len(test_dataset) <= 0:
        return None

    print(f"Holdout test (time extrapolation): {test_start.strftime('%Y-%m-%d')} ~ {test_end.strftime('%Y-%m-%d')}, {len(test_dataset)} samples")
    return DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


def train():
    """主训练函数"""
    global _RUN_START_TIME
    _RUN_START_TIME = datetime.now()

    f = Figlet(font='5lineoblique')
    print(colored(f.renderText('PRC'), 'red'))
    f = Figlet(font='digital')
    print(colored(f.renderText("Multi-Step Forecasting"), 'blue'))
    print()
    
    parser = argparse.ArgumentParser(description='PRC Thermospheric Density Forecasting')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='data/merged_datasets/satellites_data_swarm_c_10xresample_omni.csv')
    parser.add_argument('--satellite', type=str, default='swarm_c')
    parser.add_argument('--start_date', type=str, default='2014-01-01',
                        help='训练数据开始日期（约一个太阳周期 11 年）')
    parser.add_argument('--end_date', type=str, default='2024-12-31',
                        help='训练数据结束日期')
    
    # ========== 序列与数据参数 (通用) ==========
    parser.add_argument('--seq_len', type=int, default=288, help='输入序列长度 (降采样后), 例如288=24小时@5分钟分辨率')
    parser.add_argument('--pred_len', type=int, default=288, help='预测步数 (多步预测), 例如12=1小时@5分钟分辨率')
    parser.add_argument('--downsample', type=int, default=1, help='降采样因子 (swarm原始30s分辨率: 10=5分钟, 5=2.5分钟)')
    parser.add_argument('--train_type', type=str, default='log_exp_residual', choices=['log_density', 'log_exp_residual'],
                        help='训练类型: log_density直接预测密度(强制学习), log_exp_residual预测残差')

    # ========== 模型选择 ==========
    parser.add_argument('--model_type', type=str, default='patchtst',
                        choices=['informer', 'autoformer', 'lstm', 'transformer', 'tft', 'patchtst'],
                        help='模型类型: informer(长序列高效), autoformer(周期分解), lstm(基线), transformer(基线), tft(标准TFT), patchtst(PatchTST)')
    
    # ========== Transformer通用参数 (Informer/Autoformer/Transformer共用) ==========
    trans_group = parser.add_argument_group('Transformer通用参数', 'd_model/n_heads/e_layers/d_ff/dropout 适用于 Informer/Autoformer/Transformer')
    trans_group.add_argument('--d_model', type=int, default=64,help='模型维度/嵌入维度 (推荐: 128-512)')
    trans_group.add_argument('--n_heads', type=int, default=4,help='注意力头数 (需整除d_model)')
    trans_group.add_argument('--e_layers', type=int, default=2, help='Encoder层数 (推荐: 2-4层)')
    trans_group.add_argument('--d_ff', type=int, default=512,help='前馈网络维度 (推荐: d_model的2-4倍)')
    trans_group.add_argument('--dropout', type=float, default=0.3,help='Dropout率 (通用)')
    
    # ========== Informer专属参数 ==========
    informer_group = parser.add_argument_group('Informer专属参数','仅当 --model_type=informer 时生效')
    informer_group.add_argument('--factor', type=int, default=5, help='ProbSparse采样因子 (越大越稀疏,推荐3-10)')
    informer_group.add_argument('--d_layers', type=int, default=1, help='Decoder层数')
    
    # ========== Autoformer专属参数 ==========
    autoformer_group = parser.add_argument_group('Autoformer专属参数', '仅当 --model_type=autoformer 时生效')
    autoformer_group.add_argument('--moving_avg', type=str, default='25',
                                  help='序列分解移动平均核大小 (时间步数, 5分钟分辨率时，"25"对应125分钟窗口),'
                                       '支持多尺度输入如 "18,72,288"')
    
    # ========== TFT / PatchTST专属参数 ==========
    tft_group = parser.add_argument_group('TFT/PatchTST专属参数', 'tft 与 patchtst 生效')
    tft_group.add_argument('--use_variable_selection', type=lambda x: x.lower() in ['true', '1', 'yes'], default=True,
                           help='TFT 是否使用变量选择网络VSN (默认True, 设为False可加速)')
    tft_group.add_argument('--patch_len', type=int, default=16,
                           help='PatchTST 的 patch 长度 (默认16, 推荐12-24)')
    tft_group.add_argument('--patch_stride', type=int, default=8,
                           help='PatchTST 的 patch 步长 (默认8, 推荐patch_len的一半)')
    
    # ========== LSTM专属参数 ==========
    lstm_group = parser.add_argument_group('LSTM专属参数', '仅当 --model_type=lstm 时生效')
    lstm_group.add_argument('--lstm_hidden', type=int, default=128, help='LSTM隐藏层维度')
    lstm_group.add_argument('--lstm_layers', type=int, default=2,help='LSTM层数')
    
    # 训练参数
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集比例 (当k_folds=1时使用)')
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值，验证损失不改善的epoch数')
    parser.add_argument('--scheduler_patience', type=int, default=5, help='学习率调度器耐心值')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='测试集比例 (当k_folds=1时使用)')
    parser.add_argument('--k_folds', type=int, default=1, help='K折时间交叉验证折数 (1=不使用交叉验证, 3/5=多折验证)')
    parser.add_argument('--test_gap_days', type=int, default=180, help='时间外推测试：测试集起点相对 end_date 的天数（0=关闭，保持原末尾划分）')
    parser.add_argument('--val_start_date', type=str, default='', help='显式验证集开始日期（如 2023-01-01；空=按 val_ratio 比例划分）')
    parser.add_argument('--val_end_date', type=str, default='', help='显式验证集结束日期（与 val_start_date 成对使用）')
    parser.add_argument('--test_start_date', type=str, default='', help='显式测试集开始日期（如 2024-01-01；空=按 test_ratio/test_gap_days 划分）')
    parser.add_argument('--test_end_date', type=str, default='', help='显式测试集结束日期（与 test_start_date 成对使用）')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str, default='models/using')
    parser.add_argument('--wandb_active', action='store_true')
    parser.add_argument('--run_name', type=str, default='')
    
    args = parser.parse_args()
    
    # 解析多尺度移动平均核参数
    if hasattr(args, 'moving_avg'):
        if isinstance(args.moving_avg, int):
            args.kernel_sizes = [args.moving_avg]  # 单尺度
        else:
            args.kernel_sizes = [int(k.strip()) for k in str(args.moving_avg).split(',')]
    
    print('Arguments:\n{}\n'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(args), depth=2, width=1)
    print()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    if args.wandb_active:
        wandb.init(project='prc-density', config=vars(args),
                   dir=_PROJECT_ROOT,
                   mode="offline")
        if args.run_name:
            wandb.run.name = args.run_name
    else:
        print('Wandb disabled (--wandb_active not set)')
    
    print(f"\nLoading dataset with downsample={args.downsample}, seq_len={args.seq_len}, pred_len={args.pred_len}")
    dataset = SatelliteDatasetMultiStep(
        csv_path=args.data_path,
        satellite=args.satellite,
        start_date=args.start_date,
        end_date=args.end_date,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        downsample=args.downsample,
        normalize=True
    )
    
    # 定义模型参数（使用实际有效的特征列数）
    effective_input_size = len(dataset.effective_feature_columns)
    print(f"Model input size: {effective_input_size}")
    model_kwargs = {
        'input_size': effective_input_size,
        'future_cond_size': len(dataset.future_condition_columns),
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'dropout': args.dropout,
        'pred_len': args.pred_len
    }
    
    # 数据划分：支持K折时间交叉验证
    if args.k_folds > 1:
        # K折时间交叉验证：将数据分成k份，每份轮流做验证集
        print(f"\n{'='*70}")
        print(f"K-Fold Time Cross Validation: {args.k_folds} folds")
        print(f"{'='*70}")
        
        fold_size = len(dataset) // args.k_folds
        all_fold_results = []
        
        for fold_idx in range(args.k_folds):
            print(f"\n--- Fold {fold_idx + 1}/{args.k_folds} ---")
            
            # 计算当前fold的验证集范围
            val_start = fold_idx * fold_size
            val_end = min(val_start + fold_size, len(dataset))
            
            # 训练集 = 验证集之前的所有数据（保持时间顺序）
            train_indices = list(range(0, val_start))
            val_indices = list(range(val_start, val_end))
            
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
            
            print(f"Split: train=[0:{len(train_indices)}], val=[{val_start}:{val_end}]")
            print(f"Samples: train={len(train_dataset)}, val={len(val_dataset)}")
            
            # 为当前fold创建数据加载器
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            
            # 定义损失函数
            criterion = nn.MSELoss()
            
            # 为当前fold创建新模型
            if args.model_type == 'informer':
                fold_model = InformerDensityPredictor(
                    **model_kwargs,
                    e_layers=args.e_layers,
                    d_ff=args.d_ff,
                    factor=args.factor,
                    d_layers=args.d_layers
                ).to(device)
            elif args.model_type == 'lstm':
                fold_model = LSTMMultiStep(model_kwargs['input_size'], model_kwargs['future_cond_size'], args.lstm_hidden,
                                          args.lstm_layers, args.dropout, args.pred_len).to(device)
            elif args.model_type == 'transformer':
                fold_model = TransformerMultiStep(model_kwargs['input_size'], model_kwargs['future_cond_size'], args.d_model, args.n_heads,
                                                 args.e_layers, args.dropout, args.pred_len).to(device)
            elif args.model_type == 'tft':
                fold_model = TemporalFusionTransformer(
                    input_size=model_kwargs['input_size'],
                    future_cond_size=model_kwargs['future_cond_size'],
                    seq_len=args.seq_len,
                    pred_len=args.pred_len,
                    d_model=args.d_model,
                    n_heads=args.n_heads,
                    num_layers=args.e_layers,
                    dropout=args.dropout,
                    lstm_hidden=getattr(args, 'lstm_hidden', 128),
                    lstm_layers=getattr(args, 'lstm_layers', 2),
                    use_variable_selection=getattr(args, 'use_variable_selection', True)
                ).to(device)
            elif args.model_type == 'patchtst':
                fold_model = PatchTST(
                    input_size=model_kwargs['input_size'],
                    future_cond_size=model_kwargs['future_cond_size'],
                    seq_len=args.seq_len,
                    pred_len=args.pred_len,
                    patch_len=getattr(args, 'patch_len', 16),
                    stride=getattr(args, 'patch_stride', 8),
                    d_model=args.d_model,
                    n_heads=args.n_heads,
                    num_layers=args.e_layers,
                    dropout=args.dropout
                ).to(device)
            else:
                raise ValueError(f"Unknown model: {args.model_type}")
            
            # 为当前fold训练
            fold_best_val_loss, fold_results = train_single_fold(
                fold_model, train_loader, val_loader, None, criterion, device,
                args, dataset, fold_idx, model_kwargs
            )
            all_fold_results.append(fold_results)
        
        # 汇总K折结果
        print(f"\n{'='*70}")
        print("K-Fold Cross Validation Summary")
        print(f"{'='*70}")
        avg_val_mape = np.mean([r['best_val_mape'] for r in all_fold_results])
        avg_nrlmsise_mape = np.mean([r['best_nrlmsise_mape'] for r in all_fold_results])
        print(f"Average Best Val MAPE: {avg_val_mape:.2f}%")
        print(f"Average NRLMSISE MAPE: {avg_nrlmsise_mape:.2f}%")
        
        # 保存汇总结果
        summary_path = os.path.join(args.save_dir, 
            f"{args.model_type}_{args.satellite}_kfold{args.k_folds}_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"K-Fold ({args.k_folds}) Cross Validation Results\n")
            f.write(f"Average Best Val MAPE: {avg_val_mape:.2f}%\n")
            f.write(f"Average NRLMSISE MAPE: {avg_nrlmsise_mape:.2f}%\n\n")
            for i, r in enumerate(all_fold_results):
                f.write(f"Fold {i+1}: Best Val MAPE={r['best_val_mape']:.2f}%, "
                       f"NRLMSISE MAPE={r['best_nrlmsise_mape']:.2f}%\n")
        print(f"Summary saved to {summary_path}")
        
    else:
        # 单折划分：默认按时间顺序 train/val（train 70% + val 20%，末尾 10% 弃用）；
        # 支持 --val_start_date/--val_end_date 显式指定验证集时段（覆盖 val_ratio）
        if args.val_start_date and args.val_end_date:
            val_start_dt = pd.to_datetime(args.val_start_date)
            n_samples = len(dataset)
            # 每个样本 idx 覆盖行 [idx, idx+seq_len+pred_len)，预报结束行号 = idx+seq_len+pred_len-1
            sample_ends = dataset.dates[args.seq_len + args.pred_len - 1:
                                        args.seq_len + args.pred_len - 1 + n_samples]
            # 训练集 = 预报窗口完全落在验证集起始之前的样本，避免时序泄漏
            train_indices = np.where(sample_ends < np.datetime64(val_start_dt))[0].tolist()
            train_dataset = Subset(dataset, train_indices)
            # 验证集：单独按显式时段加载，复用训练集归一化参数
            val_dataset = SatelliteDatasetMultiStep(
                csv_path=args.data_path,
                satellite=args.satellite,
                start_date=args.val_start_date,
                end_date=args.val_end_date,
                seq_len=args.seq_len,
                pred_len=args.pred_len,
                downsample=args.downsample,
                normalize=True,
                norm_stats=_get_norm_stats(dataset),
            )
            print(f"显式验证集: {args.val_start_date} ~ {args.val_end_date}, {len(val_dataset)} samples")
            print(f"训练集: {len(train_dataset)} samples（预报窗口结束早于验证集起始）")
        else:
            test_size = int(len(dataset) * args.test_ratio)
            val_size = int(len(dataset) * args.val_ratio)
            train_size = len(dataset) - val_size - test_size

            train_indices = list(range(0, train_size))
            val_indices = list(range(train_size, train_size + val_size))

            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)

            print(f"Time-based split: train=[0:{train_size}], val=[{train_size}:{train_size + val_size}]")
            print(f"Dataset: train={len(train_dataset)}, val={len(val_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # 测试集：优先显式日期（--test_start_date/--test_end_date），其次时间外推
        # （end_date 后 gap_days 起，取训练时段 test_ratio 时长），最后回退到原末尾 test_ratio
        test_loader = None
        if args.test_start_date and args.test_end_date:
            test_loader = build_holdout_test_loader(args, dataset, args.start_date, args.end_date,
                                                    test_start_date=args.test_start_date,
                                                    test_end_date=args.test_end_date)
            if test_loader is None:
                print("Warning: 显式测试集数据不足，回退到训练时段末尾测试集划分")
        elif args.test_ratio > 0:
            if args.test_gap_days > 0:
                test_loader = build_holdout_test_loader(args, dataset, args.start_date, args.end_date)
                if test_loader is None:
                    print("Warning: 时间外推测试集数据不足，回退到训练时段末尾测试集划分")

            if test_loader is None:
                test_indices = list(range(train_size + val_size, len(dataset)))
                test_dataset = Subset(dataset, test_indices)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
                print(f"Test (in-range tail): [{train_size + val_size}:{len(dataset)}], {len(test_dataset)} samples")
        
        # 定义损失函数
        criterion = nn.MSELoss()
        
        # 创建模型并训练
        if args.model_type == 'informer':
            model = InformerDensityPredictor(
                **model_kwargs,
                e_layers=args.e_layers,
                d_ff=args.d_ff,
                factor=args.factor,
                d_layers=args.d_layers
            ).to(device)
        elif args.model_type == 'lstm':
            model = LSTMMultiStep(model_kwargs['input_size'], model_kwargs['future_cond_size'], args.lstm_hidden,
                                 args.lstm_layers, args.dropout, args.pred_len).to(device)
        elif args.model_type == 'transformer':
            model = TransformerMultiStep(model_kwargs['input_size'], model_kwargs['future_cond_size'], args.d_model, args.n_heads,
                                        args.e_layers, args.dropout, args.pred_len).to(device)
        elif args.model_type == 'tft':
            model = TemporalFusionTransformer(
                input_size=model_kwargs['input_size'],
                future_cond_size=model_kwargs['future_cond_size'],
                seq_len=args.seq_len,
                pred_len=args.pred_len,
                d_model=args.d_model,
                n_heads=args.n_heads,
                num_layers=args.e_layers,
                dropout=args.dropout,
                lstm_hidden=getattr(args, 'lstm_hidden', 128),
                lstm_layers=getattr(args, 'lstm_layers', 2),
                use_variable_selection=getattr(args, 'use_variable_selection', True)
            ).to(device)
        elif args.model_type == 'patchtst':
            model = PatchTST(
                input_size=model_kwargs['input_size'],
                future_cond_size=model_kwargs['future_cond_size'],
                seq_len=args.seq_len,
                pred_len=args.pred_len,
                patch_len=getattr(args, 'patch_len', 16),
                stride=getattr(args, 'patch_stride', 8),
                d_model=args.d_model,
                n_heads=args.n_heads,
                num_layers=args.e_layers,
                dropout=args.dropout
            ).to(device)
        else:
            raise ValueError(f"Unknown model: {args.model_type}")
        
        # 标准单折训练
        train_single_fold(model, train_loader, val_loader, test_loader, criterion, device,
                         args, dataset, fold_idx=0, model_kwargs=model_kwargs)
    
    if args.wandb_active:
        wandb.finish()


if __name__ == '__main__':
    time_start = time.time()
    train()
    print('\nTotal duration: {}'.format(time.time() - time_start))
    sys.exit(0)
