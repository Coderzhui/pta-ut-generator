# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.device_count 接口功能正确性
API 名称：torch.cuda.device_count
API 签名：torch.cuda.device_count() -> int

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 返回 int 类型                       | 已覆盖                  |
| 返回值范围       | > 0（当前环境 8 卡）                | 已覆盖                  |

未覆盖项及原因：
- 异常路径：无参数，不会失败

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


class TestCudaDeviceCount(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_returns_int(self):
        """Verify device_count returns an int."""
        result = torch.npu.device_count()
        self.assertIsInstance(result, int)

    def test_npu_positive_count(self):
        """Verify device_count is positive on NPU system."""
        result = torch.npu.device_count()
        self.assertGreater(result, 0)


if __name__ == "__main__":
    run_tests()
