# tests/conftest.py
"""共享 fixture：把主线脚本当模块加载（与 batch_test_final.py 相同的加载方式）。"""

import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


@pytest.fixture(scope="session")
def grasp_main():
    """scripts/test_multi_object_grasp.py 作为模块（内含全部抓取参数与辅助函数）。"""
    path = os.path.join(REPO_ROOT, "scripts", "test_multi_object_grasp.py")
    spec = importlib.util.spec_from_file_location("grasp_main_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["grasp_main_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod
