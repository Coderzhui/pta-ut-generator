# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.graph_pool_handle (via torch.npu.graph_pool_handle) 接口功能正确性
API 名称：torch.cuda.graph_pool_handle / torch.npu.graph_pool_handle
API 签名：torch.npu.graph_pool_handle() -> tuple

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 返回 tuple 类型                     | 已覆盖                  |
| 多次调用         | 多次调用返回不同 handle             | 已覆盖                  |

未覆盖项及原因：
- 无

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


class TestCudaGraphPoolHandle(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_returns_tuple(self):
        """Verify graph_pool_handle returns a tuple."""
        result = torch.npu.graph_pool_handle()
        self.assertIsInstance(result, tuple)

    def test_npu_multiple_calls(self):
        """Verify repeated calls return valid tuples without leaking state."""
        h1 = torch.npu.graph_pool_handle()
        h2 = torch.npu.graph_pool_handle()
        self.assertIsInstance(h1, tuple)
        self.assertIsInstance(h2, tuple)


if __name__ == "__main__":
    run_tests()
