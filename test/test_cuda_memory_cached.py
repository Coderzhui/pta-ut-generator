# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.memory_cached (via torch.npu.memory_cached) 接口功能正确性
API 名称：torch.cuda.memory_cached / torch.npu.memory_cached
API 签名：torch.npu.memory_cached(device=None) -> int

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 返回 int 类型                       | 已覆盖                  |
| 非负             | 返回值 >= 0                         | 已覆盖                  |
| device 参数      | 传入 device=0                      | 已覆盖                  |

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


class TestCudaMemoryCached(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_returns_int(self):
        """Verify memory_cached returns an int."""
        result = torch.npu.memory_cached()
        self.assertIsInstance(result, int)

    def test_npu_non_negative(self):
        """Verify reported value is non-negative."""
        result = torch.npu.memory_cached()
        self.assertGreaterEqual(result, 0)

    def test_npu_with_device_arg(self):
        """Verify memory_cached accepts a device index."""
        result = torch.npu.memory_cached(0)
        self.assertIsInstance(result, int)


if __name__ == "__main__":
    run_tests()
