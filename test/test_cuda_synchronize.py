# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.synchronize (via torch.npu.synchronize) 接口功能正确性
API 名称：torch.cuda.synchronize / torch.npu.synchronize
API 签名：torch.npu.synchronize(device=None) -> None

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 无参数调用返回 None                 | 已覆盖                  |
| device 参数      | 传入 device 索引                    | 已覆盖                  |
| 多次调用         | 连续调用不报错                      | 已覆盖                  |

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


class TestCudaSynchronize(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_returns_none(self):
        """Verify synchronize returns None."""
        result = torch.npu.synchronize()
        self.assertIsNone(result)

    def test_npu_with_device_arg(self):
        """Verify synchronize accepts a device index."""
        torch.npu.synchronize(0)

    def test_npu_multiple_calls(self):
        """Verify consecutive synchronize calls do not raise."""
        torch.npu.synchronize()
        torch.npu.synchronize()


if __name__ == "__main__":
    run_tests()
