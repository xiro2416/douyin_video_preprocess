"""
清理 data/ 下除 01_raw_videos 外的所有生成文件。
"""

import os
import shutil

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

KEEP = {"01_raw_videos"}

for item in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, item)
    if item in KEEP:
        print(f"跳过: {item}")
        continue
    if os.path.isfile(path):
        os.remove(path)
        print(f"删除文件: {item}")
    elif os.path.isdir(path):
        shutil.rmtree(path)
        print(f"删除目录: {item}/")

print("清理完成")
