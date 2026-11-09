"""
单星热层密度数据端到端处理脚本（一步到位）

功能：
    从下载好的原始数据（TU Delft Swarm POD txt + 太阳指数 CSV + OMNI 5 分钟数据）出发，
    一步完成：原始 txt 解析 -> 质控 -> 时间对齐重采样 -> 空间天气/OMNI 融合 ->
    NRLMSISE-00 / NRLMSIS 2.0 物理基准计算 -> 地方时特征 -> 训练就绪 CSV。

    本脚本只支持 Swarm A/B/C（version_02 POD 数据，UTC 时间，30 秒采样）。

用法：
    # 从原始 txt 一步处理（默认读取 data/tudelft_data_raw/version_02/Swarm_data 下递归搜索到的 v02 txt）
    python process_merge_sw_and_tudelft_data_simple.py --satellite swarm_c

    # 指定原始 txt 所在目录
    python process_merge_sw_and_tudelft_data_simple.py --satellite swarm_c \
        --input_raw_dir ../../data/tudelft_data_raw/version_02/Swarm_data/unzipped

    # 指定日期范围（默认全量）
    python process_merge_sw_and_tudelft_data_simple.py --satellite swarm_c \
        --date_range 2018-01-01,2024-12-31

输出：
    data/merged_datasets/satellites_data_{satellite}_{downsample}xresample_omni.csv
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

from pyfiglet import Figlet

try:
    from termcolor import colored
except ImportError:
    def colored(text, color=None):
        return text


# ============ 复用完整处理脚本中已验证的核心函数 ============
# 这些函数在 process_merge_sw_tudelft_omni.py 中已实现并经过验证：
#   - compute_msis_density：向量化分批计算 MSIS 总质量密度（比逐行循环快数十倍）
#   - load_omni_data：解析并合并 OMNI 5 分钟 ASCII 数据
#   - resample_data：按时间对齐降采样（round 到整 5 分钟再分组取首行）
#   - create_gap_flags_and_fill：间隙标记 + 前向填充
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.input_data_prep.process_merge_sw_tudelft_omni import (  # noqa: E402
    compute_msis_density,
    load_omni_data,
    resample_data,
    create_gap_flags_and_fill,
)
from scripts.input_data_prep.process_sw_indices import generate_sw_indices  # noqa: E402


GAP_THRESHOLD = 3600  # 1 小时（秒），用于数据完整性筛查

# 卫星名 -> TU Delft 文件名前缀（仅 Swarm）
SATELLITE_PREFIX = {
    'swarm_a': 'SA',
    'swarm_b': 'SB',
    'swarm_c': 'SC',
}

# Swarm POD 数据列（version_02，UTC 时间，30 秒采样）
# 原始数据行示例：
#   2023-01-01 00:00:00.000 UTC 467342.560 -127.33810469 -0.74671874 15.457 180.85501579 0.20508634E-11
#   |--- datetime ---|  alt(m)   lon(deg)      lat(deg)     lst(h)   arglat(deg)  dens_x(kg/m3)
SWARM_COLUMNS = [
    'alt [m]', 'lon [deg]', 'lat [deg]', 'lst [h]', 'arglat [deg]', 'dens_x [kg/m3]',
]


def parse_swarm_txt(filepath):
    """解析单个 Swarm POD txt 文件为 DataFrame。

    参数:
        filepath: Swarm POD txt 文件路径（TU Delft version_02 格式）。

    返回:
        pd.DataFrame，列为 Time (UTC) + SWARM_COLUMNS。
    """
    times = []
    data = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # 数据行以 ' UTC' 分隔时间与数据部分
            if ' UTC' not in line:
                continue
            datetime_part, data_part = line.split(' UTC', 1)

            vals = data_part.split()
            if len(vals) < 6:
                continue
            try:
                row = [float(v) for v in vals[:6]]
            except ValueError:
                continue
            times.append(datetime_part.strip())
            data.append(row)

    df = pd.DataFrame(data, columns=SWARM_COLUMNS)
    # 一次性向量化解析时间，避免逐行 pd.Timestamp 的开销
    df['Time (UTC)'] = pd.to_datetime(times, errors='coerce')
    df = df.dropna(subset=['Time (UTC)'])
    return df[['Time (UTC)'] + SWARM_COLUMNS].reset_index(drop=True)


def load_satellite_from_txt(raw_dir, satellite):
    """递归搜索并解析指定卫星的所有 Swarm POD txt 文件。

    参数:
        raw_dir: 原始 txt 所在根目录（递归搜索 ** 子目录）。
        satellite: 卫星名（swarm_a/b/c）。

    返回:
        合并后的 DataFrame（含规范列名），若未找到文件则返回 None。
    """
    prefix = SATELLITE_PREFIX[satellite]
    # 仅匹配 version_02（v01 为旧处理方式，列格式/质量均不同，不混用）
    pattern = os.path.join(raw_dir, '**', f'{prefix}_DNS_POD_*_v02.txt')
    files = sorted(glob.glob(pattern, recursive=True))

    # 按文件名去重：同一月份数据可能散落在多个目录（如 data/2023 与 unzipped），只读一份
    unique = {}
    for fp in files:
        unique.setdefault(os.path.basename(fp), fp)
    files = sorted(unique.values())

    if not files:
        print(f'  未在 {raw_dir} 下找到 {prefix}_DNS_POD_*_v02.txt')
        return None

    print(f'  找到 {len(files)} 个 {satellite} POD txt 文件')
    frames = []
    for fp in files:
        df = parse_swarm_txt(fp)
        if len(df) > 0:
            frames.append(df)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values('Time (UTC)').drop_duplicates(subset=['Time (UTC)']).reset_index(drop=True)
    return standardize_satellite_df(df, satellite)


def standardize_satellite_df(df, satellite):
    """将 Swarm POD txt 解析结果转换为项目规范列名。

    参数:
        df: 含 Time (UTC) 及 SWARM_COLUMNS 的 DataFrame。
        satellite: 卫星名。

    返回:
        规范列名的 DataFrame。
    """
    df = df.copy()
    df['satellite_name'] = satellite
    df['Time (UTC)'] = pd.to_datetime(df['Time (UTC)'])

    val = pd.DatetimeIndex(df['Time (UTC)'])
    df['year'] = val.year
    df['doy'] = val.day_of_year
    df['sid'] = val.hour * 3600 + val.minute * 60 + val.second + val.microsecond / 1e6

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

    # 丢弃未使用的列
    for col in ['arglat [deg]', 'lst [h]']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    return df.reset_index(drop=True)


def filter_negative_and_zero_values(df):
    """负值和零值筛查：删除密度 <= 0 的数据行。"""
    print('\n=== 负值和零值筛查 ===')
    original_count = len(df)

    density_col = 'tudelft_thermo__ground_truth_thermospheric_density__[kg/m**3]'
    if density_col in df.columns:
        valid_mask = df[density_col] > 0
        removed_count = (~valid_mask).sum()
        df = df[valid_mask].copy()
        print(f'原始数据: {original_count} 条')
        print(f'删除 <=0 的密度值: {removed_count} 条')
        print(f'剩余数据: {len(df)} 条')
    else:
        print(f'警告: 未找到密度列 {density_col}')

    return df.reset_index(drop=True)


def remove_days_with_large_gaps(df):
    """数据集完整性筛查：检测 >1 小时的间隙，剔除包含这些间隙的当日数据。"""
    print('\n=== 数据集完整性筛查 (>1小时间隙) ===')
    original_count = len(df)

    df['all__dates_datetime__'] = pd.to_datetime(df['all__dates_datetime__'])

    time_diffs = df['all__dates_datetime__'].diff().dt.total_seconds()
    large_gap_mask = time_diffs > GAP_THRESHOLD
    large_gap_count = large_gap_mask.sum()

    if large_gap_count > 0:
        print(f'检测到 {large_gap_count} 个大间隙 (>1小时)')

        gap_dates = set()
        for idx in large_gap_mask[large_gap_mask].index:
            gap_start = df.loc[idx - 1, 'all__dates_datetime__']
            gap_end = df.loc[idx, 'all__dates_datetime__']
            gap_dates.add(gap_start.date())
            gap_dates.add(gap_end.date())

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


def compute_local_time_features(df):
    """计算地方时（Local Time）的正弦/余弦编码特征。

    地方时 = UTC 秒数 / 3600 + 经度 / 15，取 24 小时模。
    """
    print("\nComputing local time features...")
    seconds_in_day = df['all__seconds_in_day__[s]'].values
    longitude = df['tudelft_thermo__longitude__[deg]'].values

    local_time_hours = (seconds_in_day / 3600.0 + longitude / 15.0) % 24.0
    df['local_time_sin'] = np.sin(2 * np.pi * local_time_hours / 24.0)
    df['local_time_cos'] = np.cos(2 * np.pi * local_time_hours / 24.0)
    print("Added local_time_sin/cos features")
    return df


def ensure_sw_indices(sw_data_dir, sw_indices_raw_dir):
    """确保太阳指数 CSV 存在，缺失时从原始数据自动生成。

    参数:
        sw_data_dir: 太阳指数 CSV 目录（celestrack_sw.csv + set_sw.csv）。
        sw_indices_raw_dir: 原始太阳指数目录（SOLFSMY.TXT / DTCFILE.TXT / SW-All.csv）。
    """
    required = ['celestrack_sw.csv', 'set_sw.csv']
    missing = [f for f in required if not os.path.exists(os.path.join(sw_data_dir, f))]
    if not missing:
        print('  太阳指数 CSV 已存在，跳过生成')
        return

    print(f'  缺少 {missing}，从原始数据自动生成太阳指数...')
    generate_sw_indices(sw_indices_raw_dir, sw_data_dir)


def process_single_satellite():
    """端到端处理单个 Swarm 卫星：原始 txt -> 训练就绪 CSV。"""
    print('Swarm POD Thermospheric Density Data Processing (One-Step)')
    f = Figlet(font='5lineoblique')
    print(colored(f.renderText('KARMAN 3.0'), 'red'))
    f = Figlet(font='digital')
    print(colored(f.renderText('Swarm POD -> Training-Ready Data'), 'blue'))

    parser = argparse.ArgumentParser(
        description='One-step Swarm POD data processing to training-ready data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--input_raw_dir', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'data', 'tudelft_data_raw', 'version_02', 'Swarm_data'),
                        help='原始 Swarm POD txt 所在根目录（递归搜索子目录）')
    parser.add_argument('--satellite', type=str, default='swarm_c',
                        help='卫星名：swarm_a / swarm_b / swarm_c')
    parser.add_argument('--input_dir_sw_data', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'data', 'sw_data'),
                        help='太阳指数 CSV 目录（celestrack_sw.csv + set_sw.csv）')
    parser.add_argument('--input_dir_sw_indices_raw', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'data', 'sw_indices_raw'),
                        help='原始太阳指数目录（缺失 CSV 时自动生成用）')
    parser.add_argument('--input_dir_omni', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'data', 'omniweb_data'),
                        help='OMNI 5 分钟数据目录')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'data', 'merged_datasets'),
                        help='输出目录')
    parser.add_argument('--downsample', type=int, default=10,
                        help='降采样因子（10 = 从 30 秒降采样到 5 分钟）')
    parser.add_argument('--date_range', type=str, default='2010-01-01,2030-12-31',
                        help='日期范围，格式: 2023-01-01,2023-12-31')

    opt = parser.parse_args()

    if opt.satellite not in SATELLITE_PREFIX:
        print(f'错误: 不支持的卫星 {opt.satellite}，仅支持 {list(SATELLITE_PREFIX.keys())}')
        sys.exit(1)

    os.makedirs(opt.output_dir, exist_ok=True)

    # ============ 1. 读取并解析原始 Swarm POD txt ============
    print(f'\n=== 读取原始 Swarm POD 数据: {opt.satellite} ===')
    df = load_satellite_from_txt(opt.input_raw_dir, opt.satellite)
    if df is None or len(df) == 0:
        print(f'错误: 未找到 {opt.satellite} 的原始 POD txt 数据，请检查 --input_raw_dir')
        sys.exit(1)
    print(f'  解析完成: {len(df)} 行, 时间范围 {df["all__dates_datetime__"].min()} ~ {df["all__dates_datetime__"].max()}')

    # ============ 2. 数据质控 ============
    print('\n=== 数据质控 ===')
    df = filter_negative_and_zero_values(df)
    df = remove_days_with_large_gaps(df)

    # ============ 3. 时间对齐重采样（默认 30s -> 5min）============
    if opt.downsample > 1:
        df = resample_data(df, opt.downsample)

    # ============ 4. 合并太阳指数（celestrack + set）============
    print('\n=== 合并太阳指数 ===')
    ensure_sw_indices(opt.input_dir_sw_data, opt.input_dir_sw_indices_raw)
    celestrack_sw = pd.read_csv(os.path.join(opt.input_dir_sw_data, 'celestrack_sw.csv'))
    celestrack_sw['all__dates_datetime__'] = pd.DatetimeIndex(celestrack_sw['all__dates_datetime__'])
    celestrack_sw.sort_values(by='all__dates_datetime__', inplace=True)

    set_sw = pd.read_csv(os.path.join(opt.input_dir_sw_data, 'set_sw.csv'))
    set_sw['all__dates_datetime__'] = pd.DatetimeIndex(set_sw['all__dates_datetime__'])
    set_sw.sort_values(by='all__dates_datetime__', inplace=True)

    sw_data = pd.merge_asof(celestrack_sw, set_sw,
                            on='all__dates_datetime__',
                            direction='backward')

    df = pd.merge_asof(df, sw_data,
                       on='all__dates_datetime__',
                       direction='backward')
    del sw_data, celestrack_sw, set_sw
    print(f'  SW 合并后: {len(df)} 行')

    # ============ 5. 合并 OMNI 数据 ============
    omni_df = load_omni_data(opt.input_dir_omni)
    if omni_df is not None and len(omni_df) > 0:
        omni_df['all__dates_datetime__'] = pd.to_datetime(omni_df['all__dates_datetime__'])
        omni_df.sort_values(by='all__dates_datetime__', inplace=True)
        df = pd.merge_asof(df, omni_df,
                           on='all__dates_datetime__',
                           direction='nearest',
                           tolerance=pd.Timedelta('3min'))
        del omni_df
        print('\n[OK] OMNI 数据融合完成')
    else:
        print('\n[WARN] 未找到 OMNI 数据，训练时将缺少 OMNI 特征列')

    # ============ 6. 计算 NRLMSISE-00 和 NRLMSIS 2.0 物理基准 ============
    print(f'\nComputing NRLMSISE-00 density for {opt.satellite}...')
    df['NRLMSISE00__thermospheric_density__[kg/m**3]'] = compute_msis_density(df, version=0)

    print(f'Computing NRLMSIS 2.0 density for {opt.satellite}...')
    df['NRLMSIS20__thermospheric_density__[kg/m**3]'] = compute_msis_density(df, version=2.0)

    df.reset_index(drop=True, inplace=True)
    df.sort_values(by='all__dates_datetime__', inplace=True)

    # ============ 7. 地方时特征 ============
    df = compute_local_time_features(df)

    # ============ 8. 清理与 NaN 处理 ============
    to_drop = [col for col in df.columns if col.startswith('Unn')]
    if to_drop:
        df.drop(to_drop, axis=1, inplace=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 只删除关键列有 NaN 的行（密度、时间、坐标）
    critical_cols = [
        'all__dates_datetime__',
        'tudelft_thermo__ground_truth_thermospheric_density__[kg/m**3]',
        'tudelft_thermo__altitude__[m]',
        'tudelft_thermo__latitude__[deg]',
        'tudelft_thermo__longitude__[deg]',
    ]
    critical_cols = [c for c in critical_cols if c in df.columns]
    df.dropna(subset=critical_cols, axis=0, inplace=True)
    print(f'[DEBUG] 删除关键列 NaN 后: {len(df)} 行')

    # 间隙标记与填充平滑（处理 OMNI 数据中的间隙）
    print('\n=== 间隙标记与填充平滑 ===')
    df = create_gap_flags_and_fill(df)

    # ============ 9. 日期范围筛选 ============
    if opt.date_range:
        start_date, end_date = opt.date_range.split(',')
        df['all__dates_datetime__'] = pd.to_datetime(df['all__dates_datetime__'])
        mask = (df['all__dates_datetime__'] >= start_date) & \
               (df['all__dates_datetime__'] <= end_date)
        df = df[mask].copy()
        df.reset_index(drop=True, inplace=True)
        print(f'\n日期范围筛选后: {len(df)} 条 ({start_date} 至 {end_date})')

    # ============ 10. 保存训练就绪数据 ============
    resolution_str = f'{opt.downsample}xresample' if opt.downsample > 1 else '1resample'
    output_name = f'satellites_data_{opt.satellite}_{resolution_str}_omni.csv'
    output_path = os.path.join(opt.output_dir, output_name)
    df.to_csv(output_path, index=False)
    print(f'\n[OK] 训练就绪数据已保存: {output_name} ({len(df)} 行)')
    print(f'     路径: {output_path}')

    print('\n=== 数据列汇总 ===')
    for col in sorted(df.columns):
        print(f'  - {col}')


if __name__ == '__main__':
    time_start = time.time()
    process_single_satellite()
    print(f'\nTotal duration: {time.time() - time_start:.2f}s')
    sys.exit(0)
