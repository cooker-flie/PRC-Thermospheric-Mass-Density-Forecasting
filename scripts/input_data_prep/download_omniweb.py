"""从 NASA OMNIWeb / SPDF 下载 OMNI 高分辨率太阳风与地磁数据。

支持三种模式（--mode）：
    5min : 直接从 SPDF 下载 omni_5min{year}.asc（训练链路实际使用，默认）
    1min : 直接从 SPDF 下载 omni_min{year}.asc（原始 1 分钟数据，备份用）
    cgi  : 通过 OMNIWeb cgi 接口下载 data_*.txt（旧下载方式，保留兼容）

5 分钟数据来源：
    https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/omni_5min{year}.asc
    数据格式见 data/omniweb_data_raw/hro_modified_format.txt
"""

import os
import sys
import time
import argparse
from datetime import datetime

import requests

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
sys.path.insert(0, _PROJECT_ROOT)

try:
    from pyfiglet import Figlet
except ImportError:
    Figlet = None

try:
    from termcolor import colored
except ImportError:
    def colored(text, color=None):
        return text


SPDF_BASE_URL = 'https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni'


def _banner():
    """打印启动横幅（pyfiglet 缺失时降级为普通文本）。"""
    if Figlet is not None:
        f = Figlet(font='digital')
        print(colored(f.renderText('Downloading OMNI Data'), 'blue'))
    else:
        print('=== Downloading OMNI Data ===')


def _download_file(url, dest):
    """下载单个文件到 dest，返回字节数。"""
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get(url, headers=headers, timeout=180)
    resp.raise_for_status()
    with open(dest, 'wb') as f:
        f.write(resp.content)
    return len(resp.content)


def download_omni_spdf(mode, years, output_dir, force=False):
    """从 SPDF 直接下载 OMNI 高分辨率 .asc 文件。

    参数:
        mode: '5min' 或 '1min'，决定文件名前缀与数据分辨率。
        years: 年份的可迭代对象。
        output_dir: 输出目录。
        force: 已存在文件是否强制覆盖重下。
    """
    prefix = 'omni_5min' if mode == '5min' else 'omni_min'
    os.makedirs(output_dir, exist_ok=True)

    downloaded = skipped = failed = 0
    for year in years:
        fname = f'{prefix}{year}.asc'
        local = os.path.join(output_dir, fname)
        if os.path.exists(local) and not force:
            print(f'  已存在，跳过: {fname}')
            skipped += 1
            continue
        url = f'{SPDF_BASE_URL}/{fname}'
        print(f'  下载: {fname}', flush=True)
        try:
            size = _download_file(url, local)
            downloaded += 1
            print(f'    完成 ({size / 1e6:.1f} MB)', flush=True)
        except Exception as e:
            failed += 1
            print(f'    失败: {e}', flush=True)
            if os.path.exists(local):
                os.remove(local)

    print(f'\n完成：下载 {downloaded} 个，跳过 {skipped} 个，失败 {failed} 个')


def download_omni_cgi(start_date, end_date, output_dir):
    """通过 OMNIWeb cgi 接口下载 data_*.txt（旧方式，1 分钟 HTML 包装数据）。

    参数:
        start_date: 起始日期，%Y%m%d 格式字符串。
        end_date: 结束日期，%Y%m%d 格式字符串。
        output_dir: 输出目录。
    """
    base_url = 'https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi'
    years = range(int(start_date[:4]), int(end_date[:4]) + 1)
    start_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')
    # 需要下载的变量（vars=11~42），参见：
    # https://omniweb.gsfc.nasa.gov/html/command_line_sample.txt
    fixed_params = ("activity=retrieve&res=min&spacecraft=omni_min"
                    "&vars=11&vars=12&vars=13&vars=14&vars=15&vars=16"
                    "&vars=17&vars=18&vars=19&vars=20&vars=21&vars=22"
                    "&vars=23&vars=24&vars=25&vars=26&vars=27&vars=28"
                    "&vars=29&vars=30&vars=31&vars=32&vars=33&vars=34"
                    "&vars=35&vars=36&vars=37&vars=38&vars=39&vars=40"
                    "&vars=41&vars=42&scale=Linear&ymin=&ymax=&view=0"
                    "&charsize=&xstyle=0&ystyle=0&symbol=0&symsize="
                    "&linestyle=solid&table=0&imagex=640&imagey=480&color=&back=")

    os.makedirs(output_dir, exist_ok=True)

    for year in years:
        if year == end_date.year:
            s_date = str(year) + '0101'
            e_date = str(year) + end_date.strftime('%m%d')
        elif year == start_date.year:
            s_date = start_date.strftime('%Y%m%d')
            e_date = str(year) + '1231'
        else:
            s_date = str(year) + '0101'
            e_date = str(year) + '1231'

        output_file = os.path.join(output_dir, f'data_{s_date}_{e_date}.txt')
        if os.path.exists(output_file):
            print(f'  已存在，跳过: {os.path.basename(output_file)}')
            continue

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': '*/*',
            'User-Agent': 'wget/1.21.3',
        }
        print(f'  下载: data_{s_date}_{e_date}.txt', flush=True)
        try:
            response = requests.post(
                base_url,
                headers=headers,
                data=f'{fixed_params}&start_date={s_date}&end_date={e_date}',
                timeout=120,
            )
            if response.status_code == 200:
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                print(f'    完成 ({len(response.content) / 1e6:.1f} MB)', flush=True)
            else:
                print(f'    失败: HTTP {response.status_code}', flush=True)
        except Exception as e:
            print(f'    失败: {e}', flush=True)
            if os.path.exists(output_file):
                os.remove(output_file)


def main():
    _banner()

    parser = argparse.ArgumentParser(
        description='OMNI 高分辨率太阳风/地磁数据下载',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--mode', type=str, default='5min',
                        choices=['5min', '1min', 'cgi'],
                        help='下载模式：5min=训练用 5 分钟 asc；1min=原始 1 分钟 asc；cgi=旧接口')
    parser.add_argument('--start_date', type=str, default='20100101',
                        help='起始日期 %%Y%%m%%d')
    parser.add_argument('--end_date', type=str, default='20261231',
                        help='结束日期 %%Y%%m%%d')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认按 mode 自动选择）')
    parser.add_argument('--force', action='store_true',
                        help='已存在文件强制覆盖重下（仅 5min/1min 模式）')
    opt = parser.parse_args()

    years = range(int(opt.start_date[:4]), int(opt.end_date[:4]) + 1)

    if opt.mode == 'cgi':
        output_dir = opt.output_dir or os.path.join(_PROJECT_ROOT, 'data', 'omniweb_data_raw')
        download_omni_cgi(opt.start_date, opt.end_date, output_dir)
    elif opt.mode == '1min':
        output_dir = opt.output_dir or os.path.join(_PROJECT_ROOT, 'data', 'omniweb_data_raw', 'unmodified')
        download_omni_spdf('1min', years, output_dir, opt.force)
    else:  # 默认 5min
        output_dir = opt.output_dir or os.path.join(_PROJECT_ROOT, 'data', 'omniweb_data')
        download_omni_spdf('5min', years, output_dir, opt.force)


if __name__ == '__main__':
    time_start = time.time()
    main()
    print('\nTotal duration: {:.1f}s'.format(time.time() - time_start))
