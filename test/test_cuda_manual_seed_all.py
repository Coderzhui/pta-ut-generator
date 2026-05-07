# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.manual_seed_all 接口功能正确性
API 名称：torch.cuda.manual_seed_all
API 签名：torch.cuda.manual_seed_all(seed) -> None

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 所有设备设置种子返回 None           | 已覆盖                  |
| 参数类型         | int 类型种子                        | 已覆盖                  |
| 边界值           | seed=0                              | 已覆盖                  |

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


class TestCudaManualSeedAll(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_returns_none(self):
        """Verify manual_seed_all returns None on success."""
        result = torch.cuda.manual_seed_all(42)
        self.assertIsNone(result)

    def test_npu_seed_zero(self):
        """Verify seed=0 (boundary value) does not raise."""
        torch.cuda.manual_seed_all(0)

    def test_npu_large_seed(self):
        """Verify large seed value (2**31-1) does not raise."""
        torch.cuda.manual_seed_all(2**31 - 1)


if __name__ == "__main__":
    run_tests()
