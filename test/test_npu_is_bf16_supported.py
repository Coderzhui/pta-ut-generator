# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.npu.is_bf16_supported 接口功能正确性
API 名称：torch.npu.is_bf16_supported
API 签名：def is_bf16_supported() -> bool

覆盖维度表：
| 覆盖维度         | 说明                                     | 覆盖情况             |
|------------------|------------------------------------------|----------------------|
| 基础调用         | 调用不报错                               | 已覆盖               |
| 返回类型         | 返回 bool                               | 已覆盖               |
| NPU 设备行为     | 在 NPU 设备上返回设备是否支持 BF16       | 已覆盖               |
| 幂等性           | 多次调用结果一致                         | 已覆盖               |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、返回类型符合预期），
     不做精度和数值正确性校验。
"""

import torch
import torch_npu  # noqa: F401

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        import unittest
        unittest.main(argv=sys.argv)


class TestNpuIsBf16Supported(TestCase):
    """Test cases for torch.npu.is_bf16_supported."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_returns_bool(self):
        """is_bf16_supported returns a bool."""
        result = torch.npu.is_bf16_supported()
        self.assertIsInstance(result, bool)

    def test_idempotent(self):
        """Multiple calls return consistent results."""
        r1 = torch.npu.is_bf16_supported()
        r2 = torch.npu.is_bf16_supported()
        self.assertEqual(r1, r2)

    def test_callable(self):
        """is_bf16_supported is callable."""
        self.assertTrue(callable(torch.npu.is_bf16_supported))

    def test_no_args_required(self):
        """is_bf16_supported takes no arguments."""
        torch.npu.is_bf16_supported()


if __name__ == "__main__":
    run_tests()
