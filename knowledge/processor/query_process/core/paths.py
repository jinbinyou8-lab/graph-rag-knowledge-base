"""查询流程路径配置。

与 import_process/core/paths.py 保持一致的相对目录约定：
所有路径都以本模块所在包（processor/query_process）为根目录，
这样整个 query_process 目录是可整体搬迁的自包含单元。
"""

import os

# 本流程包根目录： .../knowledge/processor/query_process
QUERY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 本地文件存储基础目录
LOCAL_BASE_DIR = os.path.join(QUERY_ROOT, "temp_data")

# 前端页面静态资源目录
FRONT_PAGE_DIR = os.path.join(QUERY_ROOT, "front")


def get_local_base_dir() -> str:
    """获取本地文件存储基础目录"""
    return LOCAL_BASE_DIR


def get_front_page_dir() -> str:
    """获取前端静态页面目录"""
    return FRONT_PAGE_DIR
