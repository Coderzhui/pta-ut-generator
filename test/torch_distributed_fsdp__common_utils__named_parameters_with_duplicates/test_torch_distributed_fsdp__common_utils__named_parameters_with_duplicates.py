# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.fsdp._common_utils._named_parameters_with_duplicates
API 名称：torch.distributed.fsdp._common_utils._named_parameters_with_duplicates
API 签名：_named_parameters_with_duplicates(module: nn.Module, **kwargs) -> list[tuple[str, Parameter]]

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | N/A                                                          | N/A                                            |
| 枚举选项         | N/A                                                          | N/A                                            |
| 参数类型         | nn.Module；kwargs 传 prefix 等                               | 已覆盖                                         |
| 传参与不传参     | 仅 module；带 prefix                                         | 已覆盖                                         |
| 等价类/边界值    | 简单 Linear；嵌套 Sequential                                 | 已覆盖                                         |
| 正常传参场景     | 返回 list，元素为 (str, Parameter)                           | 已覆盖                                         |
| 异常传参场景     | remove_duplicate 在 kwargs                                   | 已覆盖 AssertionError                          |

未覆盖项及原因：
- 无

注意：仅验证结构与命名前缀，不做梯度/数值校验。
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

from torch.distributed.fsdp._common_utils import _named_parameters_with_duplicates


class TestNamedParametersWithDuplicates(TestCase):
    def test_linear_lists_params(self):
        m = nn.Linear(3, 4)
        pairs = _named_parameters_with_duplicates(m)
        self.assertIsInstance(pairs, list)
        self.assertTrue(all(isinstance(n, str) for n, _ in pairs))
        names = [n for n, _ in pairs]
        self.assertIn("weight", names)

    def test_sequential_recurse_false(self):
        m = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        pairs = _named_parameters_with_duplicates(m, recurse=False)
        self.assertIsInstance(pairs, list)

    def test_prefix_filter(self):
        m = nn.Sequential(nn.Linear(1, 1))
        pairs = _named_parameters_with_duplicates(m, prefix="0.")
        self.assertGreaterEqual(len(pairs), 1)
        for n, _ in pairs:
            self.assertTrue(n.startswith("0."))

    def test_remove_duplicate_forbidden(self):
        m = nn.Linear(1, 1)
        with self.assertRaises(AssertionError):
            _named_parameters_with_duplicates(m, remove_duplicate=True)


if __name__ == "__main__":
    run_tests()
