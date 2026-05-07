# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.max_memory_reserved 接口功能正确性
API 名称：torch.cuda.max_memory_reserved
API 签名：torch.cuda.max_memory_reserved(device=None) -> int

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 返回 int 类型                       | 已覆盖                  |
| 非负             | 返回值 >= 0                         | 已覆盖                  |
| >= memory_reserved | >= 当前保留值                     | 已覆盖                  |

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


class TestCudaMaxMemoryReserved(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_returns_int(self):
        """Verify max_memory_reserved returns an int."""
        result = torch.cuda.max_memory_reserved()
        self.assertIsInstance(result, int)

    def test_npu_non_negative(self):
        """Verify reported value is non-negative."""
        result = torch.cuda.max_memory_reserved()
        self.assertGreaterEqual(result, 0)

    def test_npu_gte_memory_reserved(self):
        """Verify peak reserved memory >= current memory_reserved."""
        max_val = torch.cuda.max_memory_reserved()
        cur_val = torch.cuda.memory_reserved()
        self.assertGreaterEqual(max_val, cur_val)


if __name__ == "__main__":
    run_tests()
