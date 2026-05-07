# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.utils._get_root_modules（list 版本）
API 名称：torch.distributed.utils._get_root_modules
API 签名：_get_root_modules(modules: list[nn.Module]) -> list[nn.Module]

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | N/A                                                          | N/A                                            |
| 枚举选项         | N/A                                                          | N/A                                            |
| 参数类型         | list[Module]                                                 | 已覆盖                                         |
| 传参与不传参     | N/A                                                          | N/A                                            |
| 等价类/边界值    | 全根；父子同列表只保留根                                     | 已覆盖                                         |
| 正常传参场景     | 返回 list 且元素为 Module                                    | 已覆盖                                         |
| 异常传参场景     | N/A                                                          | 未覆盖                                         |

未覆盖项及原因：
- 无

注意：纯 Python 结构断言，无数值校验。
"""

import unittest

import torch.nn as nn

try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    pass

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

from torch.distributed.utils import _get_root_modules


class TestGetRootModules(TestCase):
    def test_all_roots(self):
        a = nn.Linear(1, 1)
        b = nn.Linear(1, 1)
        roots = _get_root_modules([a, b])
        self.assertEqual(len(roots), 2)
        self.assertIn(a, roots)
        self.assertIn(b, roots)

    def test_parent_child_returns_root_only(self):
        child = nn.Linear(2, 2)
        parent = nn.Sequential(child)
        roots = _get_root_modules([parent, child])
        self.assertEqual(len(roots), 1)
        self.assertIn(parent, roots)

    def test_single_module(self):
        m = nn.Conv2d(1, 1, 1)
        self.assertEqual(_get_root_modules([m]), [m])


if __name__ == "__main__":
    run_tests()
