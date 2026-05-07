# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.make_graphed_callables (via torch.npu.make_graphed_callables) 接口功能正确性
API 名称：torch.cuda.make_graphed_callables / torch.npu.make_graphed_callables
API 签名：torch.npu.make_graphed_callables(callables, sample_inputs, ...)

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 存在于 torch.npu 命名空间           | 已覆盖                  |

未覆盖项及原因：
- 实际调用需复杂 sample_inputs 和 graph capture 环境

注意：本测试仅验证功能正确性，不做精度和数值正确性校验。
"""
import torch
import torch_npu  # noqa: F401

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    import unittest
    from unittest import TestCase
    def run_tests():
        unittest.main(argv=sys.argv)


class TestCudaMakeGraphedCallables(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_make_graphed_callables_callable(self):
        """Verify make_graphed_callables is a callable in torch.npu namespace."""
        self.assertTrue(callable(torch.npu.make_graphed_callables))


if __name__ == "__main__":
    run_tests()
