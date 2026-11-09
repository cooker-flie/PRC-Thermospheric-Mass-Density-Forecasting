"""
融合 SW、TU Delft 和 OMNI 数据的脚本
基于 process_merge_sw_and_tudelft_data_simple.py 和 process_omniweb_data.py
"""
import argparse
import time
import sys
from datetime import datetime, timedelta

from pyfiglet import Figlet
import pandas as pd

# 尝试导入termcolor，失败时使用回退
try:
    from termcolor import colored
except ImportError:
    def colored(text, color=None):
        return text
import os
import numpy as np
import glob

# Windows 控制台默认 GBK 编码，print 含 ✓/✗ 等 Unicode 字符时会抛 UnicodeEncodeError。
# 将 stdout/stderr 的编码错误策略改为 replace（非法字符输出为 ?），避免处理中断。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors='replace')
    except (AttributeError, ValueError):
        pass

import pymsis

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 尝试导入现有的 OMNI 处理函数
try:
    from scripts.input_data_prep.process_omniweb_data import read_omni_file, parse_omni_file
except ImportError:
    # 如果无法导入，定义基本解析函数
    def read_omni_file(filepath):
        """读取 OMNI 5分钟分辨率 ASCII 文件"""
        # OMNI 数据格式：固定宽度列
        # 参考 hro_modified_format.txt
        # 前导行是头部信息，以空格开头或包含文字
        data_lines = []
        with open(filepath, 'r') as f:
            for line in f:
                # 跳过头部注释行
                if line.startswith('<') or line.startswith('>'):
                    continue
                # 检查是否是数据行（通常以年份开头，如 2010）
                parts = line.strip().split()
                if len(parts) > 0 and parts[0].isdigit() and len(parts[0]) == 4:
                    data_lines.append(line)
        return data_lines
    
    def parse_omni_file(filepath):
        """解析 OMNI 数据文件为 DataFrame"""
        # OMNI 5分钟数据列映射（根据实际49列数据）
        # 数据格式: Year DOY Hour Min ID_IMF ID_SW ...（空格分隔）
        # 注意：DST不在5分钟文件中，需单独下载1小时数据
        # OMNI物理参数筛选：剔除1-13号参数（时间/标识/质量标志）和航天器位置
        # 保留：IMF、太阳风等离子体、导出参数、地磁指数
        # 49列OMNI数据映射（基于实际line[14:]后split的45个parts结构）
        # line[14:]跳过时间列后，parts[0]对应IMF_ID
        # 实际45个parts：标识符9 + IMF/RMS 8 + 太阳风10 + 位置6 + 地磁7 + 比率2 + GOES3
        omni_columns = {
            # 索引0-8: 标识符和质量参数（跳过不解析）
            # IMF参数（索引9-16，共8列，含2列RMS）
            9: 'mag_avg_field_vector',   # Field magnitude average, nT
            10: 'bx_gse',                # Bx, nT (GSE, GSM)
            11: 'by_gse',                # By, nT (GSE)
            12: 'bz_gse',                # Bz, nT (GSE)
            13: 'by_gsm',                # By, nT (GSM)
            14: 'bz_gsm',                # Bz, nT (GSM)
            15: 'rms_sd_b',              # RMS SD B scalar
            16: 'rms_sd_field_vector',   # RMS SD field vector
            # 太阳风等离子体（索引17-26，共10列）
            17: 'flow_speed',            # Flow speed, km/s
            18: 'vx',                     # Vx Velocity, km/s, GSE
            19: 'vy',                     # Vy Velocity, km/s, GSE
            20: 'vz',                     # Vz Velocity, km/s, GSE
            21: 'proton_density',         # Proton Density, n/cc
            22: 'temperature',            # Temperature, K
            23: 'flow_pressure',           # Flow pressure, nPa
            24: 'electric_field',         # Electric field, mV/m
            25: 'plasma_beta',            # Plasma beta
            26: 'alfven_mach_number',      # Alfven mach number
            # 索引27-32: SC和BSN位置（6列，跳过不解析）
            # 地磁指数（索引33-39，共7列）
            33: 'ae',                     # AE-index, nT
            34: 'al',                     # AL-index, nT
            35: 'au',                     # AU-index, nT
            36: 'sym_d',                  # SYM/D index, nT
            37: 'sym_h',                  # SYM/H index, nT
            38: 'asym_d',                 # ASY/D index, nT
            39: 'asym_h',                 # ASY/H index, nT
            # 索引40-41: na_np_ratio和mach_number已移除（缺失率38.98%和13.44%）
            # 索引42-44: GOES通量（3列，暂不解析）
        }
        
        # 读取数据
        data = []
        line_count = 0
        valid_count = 0
        header_skipped = False
        with open(filepath, 'r') as f:
            for line in f:
                line_count += 1
                if line.startswith('<') or line.startswith('>'):
                    continue
                # 尝试固定宽度解析（OMNI标准格式）
                line = line.rstrip('\n')
                if len(line) < 100:  # 数据行至少100字符
                    continue
                
                # 跳过头部行（以非数字开头或年份不是4位数字）
                year_str = line[0:4].strip()
                if not year_str or not year_str.isdigit() or len(year_str) != 4:
                    continue
                    
                try:
                    # 固定宽度解析（参考hro_modified_format.txt）
                    # Format: (2I4,4I3...) = Year(I4), Day(I4), Hour(I3), Minute(I3)
                    # Positions: Year(0-3), Day(4-7), Hour(8-10), Minute(11-13)
                    year = int(year_str)
                    doy = int(line[4:8].strip())   # 4 chars, not 3
                    hour = int(line[8:11].strip())  # 3 chars, not 2
                    minute = int(line[11:14].strip())  # 3 chars, not 2
                    
                    # 计算具体日期
                    date = datetime(year, 1, 1) + timedelta(days=doy-1, hours=hour, minutes=minute)
                    
                    # 其余列按空白分割（从第14位开始，跳过14字符的时间字段）
                    remaining = line[14:].strip()
                    parts = remaining.split()
                    
                    row = {'datetime': date}
                    for idx, col_name in omni_columns.items():
                        if idx < len(parts):
                            try:
                                raw_str = parts[idx].strip()
                                val = float(raw_str)
                                # 检测无效值：去掉符号和小数点后是否全为9
                                digits_only = raw_str.lstrip('-+').replace('.', '').replace('e-', '').replace('e+', '').replace('e', '')
                                if digits_only and all(c == '9' for c in digits_only):
                                    row[col_name] = np.nan
                                # 同时检测常见填充值格式
                                elif val in [99.0, 999.0, 9999.0, 99999.0, 999999.0, 9999999.0,
                                           99.99, 999.99, 9999.99, 99999.99,
                                           9.9, 99.9, 999.9, 9999.9]:
                                    row[col_name] = np.nan
                                else:
                                    row[col_name] = val
                            except:
                                row[col_name] = np.nan
                    data.append(row)
                    valid_count += 1
                except Exception as e:
                    if line_count <= 5:  # 只显示前5行错误
                        print(f'    [DEBUG] Line {line_count} skip: {str(e)[:50]}')
                    continue
        
        print(f'    解析完成: {valid_count}/{line_count} 行有效')
        
        if not data:
            return None
            
        df = pd.DataFrame(data)
        
        # 调试输出
        print(f'    解析列: {list(df.columns)}')
        if 'proton_density' in df.columns and len(df) > 0:
            print(f'    样本值: proton_density={df["proton_density"].iloc[0]:.2f}, flow_speed={df.get("flow_speed", pd.Series([np.nan])).iloc[0]:.1f}')
        if 'ae' in df.columns and len(df) > 0:
            print(f'    指数样本: ae={df["ae"].iloc[0]:.1f}, al={df.get("al", pd.Series([np.nan])).iloc[0]:.1f}, sym_h={df.get("sym_h", pd.Series([np.nan])).iloc[0]:.1f}')
        
        return df


# 原有函数保持
GAP_THRESHOLD = 3600  # 1小时（秒）


def create_gap_flags_and_fill(df):
    """
    间隙标记与填充平滑
    参照 process_omniweb_data.py 的实现
    1. 标记间隙（时间间隔 > 5分钟的数据点）
    2. 前向填充（ffill）处理缺失值
    """
    if df.empty:
        return df
    
    # 确保时间列已排序
    df = df.sort_values('all__dates_datetime__').reset_index(drop=True)
    
    # 计算时间间隔
    time_diffs = df['all__dates_datetime__'].diff().dt.total_seconds()
    
    # 5分钟 = 300秒，标记间隙（包括数据缺失导致的间断）
    gap_threshold = 300  # 5分钟
    gap_flags = time_diffs > gap_threshold
    
    # 添加间隙标记列
    df['source__gaps_flag__'] = gap_flags.astype(int)
    
    # 识别需要填充的列（OMNI相关列）
    omni_cols = [col for col in df.columns if col.startswith('omni__')]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_fill = [col for col in numeric_cols if col not in ['source__gaps_flag__', 'all__seconds_in_day__[s]', 'all__day_of_year__[d]', 'all__year__[y]']]
    
    # 前向填充（限3个连续缺失值）
    filled_count = 0
    for col in cols_to_fill:
        na_before = df[col].isna().sum()
        df[col] = df[col].ffill(limit=3)
        na_after = df[col].isna().sum()
        filled_count += (na_before - na_after)
    
    print(f'  间隙标记: {gap_flags.sum()} 个间断点')
    print(f'  填充处理: {filled_count} 个缺失值已填充（限3个连续值）')
    
    return df


def filter_negative_and_zero_values(df):
    """负值和零值筛查：删除密度 <= 0 的数据行"""
    print('\n=== 负值和零值筛查 ===')
    original_count = len(df)
    
    # 检查密度相关列
    density_cols = [c for c in df.columns if 'density' in c.lower() or 'dens' in c.lower()]
    print(f'  找到密度相关列: {density_cols}')
    print(f'  所有列: {list(df.columns)[:10]}...')
    
    density_col = 'tudelft_thermo__ground_truth_thermospheric_density__[kg/m**3]'
    if density_col in df.columns:
        print(f'  密度列统计: min={df[density_col].min():.6e}, max={df[density_col].max():.6e}, mean={df[density_col].mean():.6e}')
        valid_mask = df[density_col] > 0
        removed_count = (~valid_mask).sum()
        df = df[valid_mask].copy()
        print(f'原始数据: {original_count} 条')
        print(f'删除 <=0 的密度值: {removed_count} 条')
        print(f'剩余数据: {len(df)} 条')
    else:
        print(f'警告: 未找到密度列 {density_col}')
        print(f'  实际列名: {list(df.columns)}')
    
    return df.reset_index(drop=True)


def remove_days_with_large_gaps(df):
    """数据集完整性筛查：检测>1小时的间隙，剔除包含这些间隙的当日数据"""
    print('\n=== 数据集完整性筛查 (>1小时间隙) ===')
    original_count = len(df)
    
    df['all__dates_datetime__'] = pd.to_datetime(df['all__dates_datetime__'])
    
    # 计算时间间隔（秒）
    time_diffs = df['all__dates_datetime__'].diff().dt.total_seconds()
    
    # 找出大于1小时的间隙
    large_gap_mask = time_diffs > GAP_THRESHOLD
    large_gap_count = large_gap_mask.sum()
    
    if large_gap_count > 0:
        print(f'检测到 {large_gap_count} 个大间隙 (>1小时)')
        
        # 获取大间隙所在的日期
        gap_dates = set()
        for idx in large_gap_mask[large_gap_mask].index:
            gap_start = df.loc[idx-1, 'all__dates_datetime__']
            gap_end = df.loc[idx, 'all__dates_datetime__']
            gap_dates.add(gap_start.date())
            gap_dates.add(gap_end.date())
        
        # 剔除包含大间隙的日期
        df['date'] = df['all__dates_datetime__'].dt.date
        df = df[~df['date'].isin(gap_dates)].copy()
        removed_count = original_count - len(df)
        
        print(f'删除 {len(gap_dates)} 个有间隙的日期')
        print(f'删除样本数: {removed_count} 条')
        print(f'剩余样本数: {len(df)} 条')
        df = df.drop(columns=['date'])
    else:
        print('未检测到大间隙 (>1小时)')
    
    return df.reset_index(drop=True)


def resample_data(df, downsample_factor=10):
    """数据重采样：降采样到5分钟整点（与OMNI对齐）"""
    from pandas import Timedelta
    
    print(f'\n=== 数据重采样 (降采样因子={downsample_factor}) ===')
    original_count = len(df)
    
    if downsample_factor > 1:
        # 计算目标分辨率（秒）
        target_seconds = downsample_factor * 30  # 默认 300秒=5分钟
        
        # 将时间 round 到最近的5分钟整点
        # 例如: 00:00:30 -> 00:00:00, 00:02:30 -> 00:00:00, 00:03:00 -> 00:05:00
        df = df.copy()
        df['all__dates_datetime__'] = pd.to_datetime(df['all__dates_datetime__'])
        
        # 使用 round 到5分钟
        df['time_rounded'] = df['all__dates_datetime__'].dt.round(f'{target_seconds}s')
        
        # 按 round 后的时间分组，取每组第一行
        df = df.groupby('time_rounded', as_index=False).first()
        
        # 更新时间列为 round 后的时间
        df['all__dates_datetime__'] = df['time_rounded']
        df.drop(columns=['time_rounded'], inplace=True)
        
        print(f'降采样后样本数: {len(df)} 条 (原始: {original_count})')
        
        # 计算时间分辨率
        if len(df) >= 2:
            time_diff = (df['all__dates_datetime__'].iloc[1] - df['all__dates_datetime__'].iloc[0]).total_seconds()
            print(f'时间分辨率: ~{time_diff/60:.1f} 分钟')
            # 显示几个样本时间点
            print(f'  样本时间点: {df["all__dates_datetime__"].iloc[:3].tolist()}')
        else:
            print('警告: 降采样后数据不足2行，无法计算时间分辨率')
    else:
        print('降采样因子=1，不进行降采样')
    
    return df.reset_index(drop=True)


def compute_msis_density(df, version=0):
    """向量化计算 MSIS 总质量密度（kg/m³）。

    version: 0 = NRLMSISE-00, 2.0 = NRLMSIS 2.0, 2.1 = NRLMSIS 2.1
    返回 numpy 数组，长度与 df 行数一致。
    """

    n = len(df)
    print(f"Computing MSIS (version={version}) for {n} rows...")

    dates = pd.to_datetime(df['all__dates_datetime__']).to_numpy()
    alts_km = df['tudelft_thermo__altitude__[m]'].to_numpy() / 1000.0
    lons = df['tudelft_thermo__longitude__[deg]'].to_numpy()
    lats = df['tudelft_thermo__latitude__[deg]'].to_numpy()

    # f107: 优先 OMNI 的 f10.7，否则用 SET 的 f107_obs，默认 150
    if 'omni__f10.7' in df.columns:
        f107 = df['omni__f10.7'].to_numpy(dtype=float)
    elif 'space_environment_technologies__f107_obs__' in df.columns:
        f107 = df['space_environment_technologies__f107_obs__'].to_numpy(dtype=float)
    else:
        f107 = np.full(n, 150.0)
    f107 = np.where(np.isnan(f107), 150.0, f107)

    # f107a: F10.7 81 天平均，缺省时用 f107
    if 'space_environment_technologies__f107_average__' in df.columns:
        f107a = df['space_environment_technologies__f107_average__'].to_numpy(dtype=float)
    else:
        f107a = f107.copy()
    f107a = np.where(np.isnan(f107a), f107, f107a)

    # ap: 优先 OMNI 的 ap，否则用 celestrack 的 ap_average，默认 15
    if 'omni__ap' in df.columns:
        ap = df['omni__ap'].to_numpy(dtype=float)
    elif 'celestrack__ap_average__' in df.columns:
        ap = df['celestrack__ap_average__'].to_numpy(dtype=float)
    else:
        ap = np.full(n, 15.0)
    ap = np.where(np.isnan(ap), 15.0, ap)

    # 分批向量化计算，避免一次性内存过大
    batch_size = 100000
    densities = np.full(n, np.nan)
    n_batches = int(np.ceil(n / batch_size))
    for i in range(n_batches):
        s = i * batch_size
        e = min(s + batch_size, n)
        ap_batch = np.repeat(ap[s:e], 7).reshape(-1, 7)
        try:
            out = pymsis.calculate(dates[s:e], lons[s:e], lats[s:e],
                                   alts_km[s:e], f107[s:e], f107a[s:e],
                                   ap_batch, version=version)
            densities[s:e] = out[:, 0]
        except Exception as ex:
            print(f"  Batch {i} failed: {ex}")

    return densities


def load_omni_data(omni_data_dir, start_year=None, end_year=None):
    """加载 OMNI 数据文件"""
    print('\n=== 加载 OMNI 数据 ===')
    
    omni_files = glob.glob(os.path.join(omni_data_dir, 'omni_5min*.asc'))
    if not omni_files:
        # 尝试其他命名模式
        omni_files = glob.glob(os.path.join(omni_data_dir, '*.asc'))
        omni_files = [f for f in omni_files if 'omni' in f.lower()]
    
    if not omni_files:
        print('警告: 未找到 OMNI 数据文件')
        return None
    
    print(f'找到 {len(omni_files)} 个 OMNI 文件')
    
    all_omni_data = []
    for filepath in sorted(omni_files):
        filename = os.path.basename(filepath)
        print(f'  读取: {filename}')
        try:
            df = parse_omni_file(filepath)
            if df is not None and len(df) > 0:
                all_omni_data.append(df)
        except Exception as e:
            print(f'    错误: {e}')
    
    if not all_omni_data:
        return None
    
    # 合并所有年份数据
    omni_df = pd.concat(all_omni_data, ignore_index=True)
    omni_df = omni_df.sort_values('datetime').reset_index(drop=True)
    
    # 重命名列为统一格式
    # 注意：已剔除1-13号参数（时间/标识/质量标志）和航天器位置参数
    omni_column_mapping = {
        'datetime': 'all__dates_datetime__',
        # IMF参数
        'mag_avg_field_vector': 'omni__mag_avg_field_vector__[nT]',
        'bx_gse': 'omni__bx_gse__[nT]',
        'by_gse': 'omni__by_gse__[nT]',
        'bz_gse': 'omni__bz_gse__[nT]',
        'by_gsm': 'omni__by_gsm__[nT]',
        'bz_gsm': 'omni__bz_gsm__[nT]',
        'rms_sd_b': 'omni__rms_sd_b__[nT]',
        'rms_sd_field_vector': 'omni__rms_sd_field_vector__[nT]',
        # 太阳风等离子体
        'flow_speed': 'omni__flow_speed__[km/s]',
        'vx': 'omni__vx__[km/s]',
        'vy': 'omni__vy__[km/s]',
        'vz': 'omni__vz__[km/s]',
        'proton_density': 'omni__proton_density__[n/cm3]',
        'temperature': 'omni__temperature__[K]',
        # 导出参数
        'flow_pressure': 'omni__flow_pressure__[nPa]',
        'electric_field': 'omni__electric_field__[mV/m]',
        'plasma_beta': 'omni__plasma_beta__',
        'alfven_mach_number': 'omni__alfven_mach_number__',
        # 地磁指数
        'ae': 'omni__ae__[nT]',
        'al': 'omni__al__[nT]',
        'au': 'omni__au__[nT]',
        'sym_d': 'omni__sym_d__[nT]',
        'sym_h': 'omni__sym_h__[nT]',
        'asym_d': 'omni__asym_d__[nT]',
        'asym_h': 'omni__asym_h__[nT]',
        # 剔除的参数：na_np_ratio(38.98%缺失), mach_number(13.44%缺失)
        # 剔除的参数：
        # - 1-13号参数：id_by_imf/sw, points_in_avg, percent_interp, timeshift, rms等
        # - 航天器位置：x/y/z_position, bsn_x/y/z
        # 'dst' 不在5分钟文件中
    }
    
    omni_df = omni_df.rename(columns={k: v for k, v in omni_column_mapping.items() if k in omni_df.columns})
    
    # 处理OMNI数据中的NaN（无限制前向填充，类似process_omniweb_data.py）
    omni_cols = [c for c in omni_df.columns if c.startswith('omni__')]
    if omni_cols:
        nan_before = omni_df[omni_cols].isna().sum().sum()
        # 按时间排序后前向填充（无限制）
        omni_df = omni_df.sort_values('all__dates_datetime__').reset_index(drop=True)
        omni_df[omni_cols] = omni_df[omni_cols].fillna(method='ffill')
        nan_after = omni_df[omni_cols].isna().sum().sum()
        print(f'    NaN处理: {nan_before} -> {nan_after} ({nan_before - nan_after} 个已填充，剩余为数据起始段NaN)')
    
    print(f'OMNI 数据加载完成: {len(omni_df)} 条记录')
    print(f'时间范围: {omni_df["all__dates_datetime__"].min()} 至 {omni_df["all__dates_datetime__"].max()}')
    
    return omni_df


def merge_all_data():
    print('Merging Satellite, TU Delft and OMNI Data')
    f = Figlet(font='5lineoblique')
    print(colored(f.renderText('KARMAN 3.0'), 'red'))
    f = Figlet(font='digital')
    print(colored(f.renderText('Merging SW + TU Delft + OMNI'), 'blue'))

    parser = argparse.ArgumentParser(
        description='Merging Space Weather, TU Delft and OMNI Data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input_dir', type=str, default='../../data/tudelft_data',
                       help='Input directory with satellite data files')
    parser.add_argument('--satellite', type=str, default='swarm_c',
                       help='Satellite to process: swarm_a/b/c, grace_a/b/c, goce, champ, or all')
    parser.add_argument('--input_dir_sw_data', type=str, default='../../data/sw_data',
                       help='Input directory of SW Indices data')
    parser.add_argument('--input_dir_omni', type=str, default='../../data/omniweb_data',
                       help='Input directory of OMNI data')
    parser.add_argument('--output_dir', type=str, default='../../data/merged_datasets',
                       help='Output directory of processed data')
    parser.add_argument('--downsample', type=int, default=10,
                       help='降采样因子（10=5分钟从30秒数据）')
    parser.add_argument('--date_range', type=str, default='2010-01-01,2030-12-31',
                       help='日期范围，格式: 2023-01-01,2023-12-31')

    opt = parser.parse_args()
    os.makedirs(opt.output_dir, exist_ok=True)

    # Load satellite data
    if opt.satellite == 'all':
        input_file = os.path.join(opt.input_dir, 'satellites_data.csv')
    else:
        input_file = os.path.join(opt.input_dir, f'{opt.satellite}_data.csv')
    
    print(f'Reading satellite data: {input_file}')
    df = pd.read_csv(input_file)
    print(f'  读取完成: {len(df)} 行, {len(df.columns)} 列')
    
    # Convert column names if using raw single satellite file
    if 'Time (UTC)' in df.columns:
        df['satellite_name'] = opt.satellite
        df['Time (UTC)'] = pd.to_datetime(df['Time (UTC)'])
        
        # Add time columns
        val = pd.DatetimeIndex(df['Time (UTC)'])
        df['year'] = val.year
        df['doy'] = val.day_of_year
        df['sid'] = val.hour * 3600 + val.minute * 60 + val.second + val.microsecond / 1e6
        
        # Rename to standard format
        df.rename(columns={
            'Time (UTC)': 'all__dates_datetime__',
            'doy': 'all__day_of_year__[d]',
            'year': 'all__year__[y]',
            'sid': 'all__seconds_in_day__[s]',
            'dens_x [kg/m3]': 'tudelft_thermo__ground_truth_thermospheric_density__[kg/m**3]',
            'lon [deg]': 'tudelft_thermo__longitude__[deg]',
            'lat [deg]': 'tudelft_thermo__latitude__[deg]',
            'alt [m]': 'tudelft_thermo__altitude__[m]',
            'satellite_name': 'tudelft_thermo__satellite__',
        }, inplace=True)
        
        # Drop unused columns if exist
        for col in ['arglat [deg]', 'lst [h]']:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
    
    df['all__dates_datetime__'] = pd.DatetimeIndex(df['all__dates_datetime__'].values)

    # 1. Space weather data (SW)
    celestrack_sw = pd.read_csv(os.path.join(opt.input_dir_sw_data, 'celestrack_sw.csv'))
    celestrack_sw['all__dates_datetime__'] = pd.DatetimeIndex(celestrack_sw['all__dates_datetime__'])
    celestrack_sw.sort_values(by='all__dates_datetime__', inplace=True)

    set_sw = pd.read_csv(os.path.join(opt.input_dir_sw_data, 'set_sw.csv'))
    set_sw['all__dates_datetime__'] = pd.DatetimeIndex(set_sw['all__dates_datetime__'])
    set_sw.sort_values(by='all__dates_datetime__', inplace=True)

    sw_data = pd.merge_asof(celestrack_sw, set_sw,
                            on='all__dates_datetime__',
                            direction='backward')

    # 2. Merge satellite with SW
    print(f'\n[DEBUG] 卫星数据合并前: {len(df)} 行, 时间范围: {df["all__dates_datetime__"].min()} ~ {df["all__dates_datetime__"].max()}')
    df_merged = pd.merge_asof(df, sw_data,
                              on='all__dates_datetime__',
                              direction='backward')
    print(f'[DEBUG] SW合并后: {len(df_merged)} 行')
    del df, sw_data, celestrack_sw

    # 3. 数据质量控制（在降采样前进行，确保数据质量）
    print('\n=== 数据质量控制（原始30秒分辨率） ===')
    
    # 3.1 负值和零值筛查
    df_merged = filter_negative_and_zero_values(df_merged)
    
    # 3.2 数据集完整性筛查（>1小时间隙剔除）
    df_merged = remove_days_with_large_gaps(df_merged)
    
    # 3.3 数据重采样（降采样到5分钟，与OMNI分辨率一致）
    if opt.downsample > 1:
        df_merged = resample_data(df_merged, opt.downsample)
    
    # 4. Load and merge OMNI data（降采样后进行，分辨率一致）
    omni_df = load_omni_data(opt.input_dir_omni)
    if omni_df is not None and len(omni_df) > 0:
        print(f'[DEBUG] OMNI数据: {len(omni_df)} 行, 时间范围: {omni_df["all__dates_datetime__"].min()} ~ {omni_df["all__dates_datetime__"].max()}')
        # 确保时间列格式一致
        omni_df['all__dates_datetime__'] = pd.to_datetime(omni_df['all__dates_datetime__'])
        omni_df.sort_values(by='all__dates_datetime__', inplace=True)
        
        # 调试：显示前几行时间样本
        print(f'  卫星样本时间: {df_merged["all__dates_datetime__"].iloc[:3].tolist()}')
        print(f'  OMNI样本时间: {omni_df["all__dates_datetime__"].iloc[:3].tolist()}')
        
        # 使用 merge_asof 按时间对齐（两边都是5分钟，nearest模式找最近匹配）
        df_merged = pd.merge_asof(df_merged, omni_df,
                                  on='all__dates_datetime__',
                                  direction='nearest',
                                  tolerance=pd.Timedelta('3min'))
        del omni_df
        
        # 检查OMNI列的NaN比例
        omni_cols = [c for c in df_merged.columns if c.startswith('omni__')]
        if omni_cols:
            nan_ratios = df_merged[omni_cols].isna().mean()
            print(f'  OMNI列NaN比例: {nan_ratios.mean()*100:.1f}%')
        
        print(f'[DEBUG] OMNI合并后: {len(df_merged)} 行')
        print('\n[✓] OMNI 数据融合完成')
    else:
        print('\n[!] 跳过 OMNI 数据融合')

    # 5. Add NRLMSISE-00 和 NRLMSIS 2.0 density（降采样后计算，速度快10倍）
    print(f"\nComputing NRLMSISE-00 density for {opt.satellite}...")
    df_merged['NRLMSISE00__thermospheric_density__[kg/m**3]'] = compute_msis_density(df_merged, version=0)

    print(f"Computing NRLMSIS 2.0 density for {opt.satellite}...")
    df_merged['NRLMSIS20__thermospheric_density__[kg/m**3]'] = compute_msis_density(df_merged, version=2.0)
    
    df_merged.reset_index(drop=True, inplace=True)
    df_merged.sort_values(by='all__dates_datetime__', inplace=True)
    
    # 计算地方时（Local Time）特征
    print("Computing local time features...")
    df_merged['seconds_in_day'] = df_merged['all__seconds_in_day__[s]'].values
    df_merged['longitude'] = df_merged['tudelft_thermo__longitude__[deg]'].values
    
    # 地方时 = UTC小时 + 经度/15（每15度经度对应1小时）
    local_time_hours = (df_merged['seconds_in_day'] / 3600.0 + df_merged['longitude'] / 15.0) % 24.0
    
    # 正弦编码（捕捉周期性）
    df_merged['local_time_sin'] = np.sin(2 * np.pi * local_time_hours / 24.0)
    df_merged['local_time_cos'] = np.cos(2 * np.pi * local_time_hours / 24.0)
    
    # 删除临时列
    df_merged.drop(columns=['seconds_in_day', 'longitude'], inplace=True)
    print(f"Added local_time_sin/cos features")

    to_drop = [col for col in df_merged.columns if col.startswith('Unn')]
    if to_drop:
        df_merged.drop(to_drop, axis=1, inplace=True)

    df_merged.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 检查NaN情况
    print(f'\n[DEBUG] NRLMSISE00计算后: {len(df_merged)} 行')
    nan_counts = df_merged.isna().sum()
    if nan_counts.sum() > 0:
        print('  NaN分布:')
        for col, count in nan_counts[nan_counts > 0].items():
            print(f'    {col}: {count} ({count/len(df_merged)*100:.1f}%)')

    # 只删除关键列有NaN的行（密度、时间、坐标）
    critical_cols = [
        'all__dates_datetime__',
        'tudelft_thermo__ground_truth_thermospheric_density__[kg/m**3]',
        'tudelft_thermo__altitude__[m]',
        'tudelft_thermo__latitude__[deg]',
        'tudelft_thermo__longitude__[deg]'
    ]
    critical_cols = [c for c in critical_cols if c in df_merged.columns]
    df_merged.dropna(subset=critical_cols, axis=0, inplace=True)
    print(f'[DEBUG] 删除关键列NaN后: {len(df_merged)} 行')
    
    # 间隙标记与填充平滑（处理OMNI数据中的间隙）
    print('\n=== 间隙标记与填充平滑 ===')
    df_merged = create_gap_flags_and_fill(df_merged)
    
    # 日期范围筛选（如果指定）
    if opt.date_range:
        start_date, end_date = opt.date_range.split(',')
        df_merged['all__dates_datetime__'] = pd.to_datetime(df_merged['all__dates_datetime__'])
        mask = (df_merged['all__dates_datetime__'] >= start_date) & \
               (df_merged['all__dates_datetime__'] <= end_date)
        df_merged = df_merged[mask].copy()
        df_merged.reset_index(drop=True, inplace=True)
        print(f'\n日期范围筛选后: {len(df_merged)} 条 ({start_date} 至 {end_date})')

    # Save full data
    resolution_str = f"{opt.downsample}xresample" if opt.downsample > 1 else "1resample"
    if opt.satellite != 'all':
        output_name = f'satellites_data_{opt.satellite}_{resolution_str}_omni.csv'
    else:
        output_name = f'satellites_data_{resolution_str}_omni.csv'
    output_path = os.path.join(opt.output_dir, output_name)
    df_merged.to_csv(output_path, index=False)
    print(f'\n[✓] 融合数据已保存: {output_name} ({len(df_merged)} rows)')
    print(f'    路径: {output_path}')
    
    # 输出列名汇总
    print('\n=== 数据列汇总 ===')
    for col in sorted(df_merged.columns):
        print(f'  - {col}')


if __name__ == '__main__':
    time_start = time.time()
    merge_all_data()
    print(f'\nTotal duration: {time.time() - time_start:.2f}s')
    sys.exit(0)
