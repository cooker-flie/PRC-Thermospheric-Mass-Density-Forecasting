# TU Delft 热层密度数据下载脚本
# 原 FTP 服务器（thermosphere.tudelft.nl FTP）已于 2025-12 关闭，
# 改为 HTTPS web 服务器：https://thermosphere.tudelft.nl/data
# 本脚本从新的 HTTPS 服务器递归下载并自动解压 zip 文件。
# 已存在的文件自动跳过，可重复运行以增量更新。

import os
import sys
import time
import argparse
import zipfile
import urllib.request
import urllib.parse
from html.parser import HTMLParser

from pyfiglet import Figlet
from termcolor import colored


class DirectoryListParser(HTMLParser):
    """解析 Apache 'Index of' 页面，提取子目录和文件链接。"""

    def __init__(self):
        super().__init__()
        self.links = []  # (name, is_directory)

    def handle_starttag(self, tag, attrs):
        if tag != 'a':
            return
        href = dict(attrs).get('href', '')
        if not href or href.startswith('?') or href.startswith('/'):
            return
        # 目录以 / 结尾
        if href.endswith('/'):
            name = href.rstrip('/')
            if name and name != '..':
                self.links.append((name, True))
        else:
            name = urllib.parse.unquote(href)
            if name and not name.startswith('.'):
                self.links.append((name, False))


def fetch_url(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def list_directory(url):
    """返回 (subdirs, files)。"""
    html = fetch_url(url).decode('utf-8', errors='replace')
    parser = DirectoryListParser()
    parser.feed(html)
    subdirs = [n for n, d in parser.links if d]
    files = [n for n, d in parser.links if not d]
    return subdirs, files


def download_directory(url, local_dir):
    """递归下载目录内容，已存在文件跳过。"""
    os.makedirs(local_dir, exist_ok=True)
    subdirs, files = list_directory(url)
    for f in files:
        local_path = os.path.join(local_dir, f)
        if os.path.exists(local_path):
            print(f"已存在，跳过: {local_path}", flush=True)
            continue
        file_url = url.rstrip('/') + '/' + urllib.parse.quote(f)
        print(f"下载: {f}", flush=True)
        try:
            data = fetch_url(file_url)
            with open(local_path, 'wb') as fh:
                fh.write(data)
            print(f"  完成 ({len(data)/1e6:.1f} MB)", flush=True)
        except Exception as e:
            print(f"  失败: {e}", flush=True)
            if os.path.exists(local_path):
                os.remove(local_path)
    for sub in subdirs:
        download_directory(url.rstrip('/') + '/' + urllib.parse.quote(sub) + '/',
                           os.path.join(local_dir, sub))


def unzip_all_files(directory):
    """解压目录及其子目录下所有 zip 文件。"""
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith('.zip'):
                continue
            file_path = os.path.join(root, file)
            extract_dir = os.path.join(root, 'unzipped')
            # 已解压则跳过
            if os.path.exists(os.path.join(extract_dir, file[:-4])):
                continue
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                print(f"解压: {file}", flush=True)
            except Exception as e:
                print(f"解压失败 {file}: {e}", flush=True)


def main():
    print('TU Delft Thermospheric Density Data Downloading')
    f = Figlet(font='5lineoblique')
    print(colored(f.renderText('KARMAN 2.0'), 'red'))
    f = Figlet(font='digital')
    print(colored(f.renderText("Downloading TU Delft Thermospheric Density Data"), 'blue'))

    parser = argparse.ArgumentParser(description='TU Delft Thermosphere Data Downloading (HTTPS)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 默认路径基于脚本位置解析，避免依赖运行时的当前工作目录
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.abspath(os.path.join(_script_dir, '..', '..'))
    parser.add_argument('--base_url', type=str,
                        default='https://thermosphere.tudelft.nl/data/data',
                        help='HTTPS 服务器数据根目录')
    parser.add_argument('--tudelft_data_dir', type=str,
                        default=os.path.join(_project_root, 'data', 'tudelft_data_raw'),
                        help='本地存储目录')
    opt = parser.parse_args()

    data_dir = os.path.abspath(opt.tudelft_data_dir)
    print(f"数据目录: {data_dir}")
    print(f"服务器: {opt.base_url}")

    # 递归下载
    download_directory(opt.base_url, data_dir)

    # 解压
    print("\n解压 zip 文件...")
    unzip_all_files(data_dir)

    print("\n全部完成。")


if __name__ == "__main__":
    time_start = time.time()
    main()
    print('\nTotal duration: {}'.format(time.time() - time_start))
    sys.exit(0)
