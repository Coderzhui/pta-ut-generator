# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.current_device (via torch.npu.current_device) 接口功能正确性
API 名称：torch.cuda.current_device / torch.npu.current_device
API 签名：torch.npu.current_device() -> int

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 返回 int 类型设备索引               | 已覆盖                  |
| 返回值范围       | 0 <= result < device_count          | 已覆盖                  |
| 设备切换后       | set_device 后 current_device 匹配   | 已覆盖                  |

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


class TestCudaCurrentDevice(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")
        self.device = torch.device(self.device_name)
        self._orig_device = torch.npu.current_device()

    def tearDown(self):
        torch.npu.set_device(self._orig_device)
        super().tearDown()

    def test_npu_returns_int(self):
        """Verify current_device returns an int."""
        result = torch.npu.current_device()
        self.assertIsInstance(result, int)

    def test_npu_returns_valid_index(self):
        """Verify current_device returns a valid device index."""
        result = torch.npu.current_device()
        self.assertGreaterEqual(result, 0)
        self.assertLess(result, torch.npu.device_count())

    def test_npu_after_set_device(self):
        """Verify current_device matches after set_device."""
        torch.npu.set_device(0)
        self.assertEqual(torch.npu.current_device(), 0)


if __name__ == "__main__":
    run_tests()
