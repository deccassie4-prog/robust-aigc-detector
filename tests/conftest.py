"""pytest 根配置：把仓库根注入 sys.path，使 tests/ 能导入 scripts/、src/、evaluation/。

直接以 `python tests/test_xxx.py` 运行时同样生效（各测试文件也保留了双模式入口）。
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
