# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.is_current_stream_capturing (via torch.npu.is_current_stream_capturing) 接口功能正确性
API 名称：torch.cuda.is_current_stream_capturing / torch.npu.is_current_stream_capturing
API 签名：torch.npu.is_current_stream_capturing() -> bool

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 返回 bool 类型                      | 已覆盖                  |
| 非捕获状态       | 默认返回 False                      | 已覆盖                  |

未覆盖项及原因：
- 捕获中状态：需要在 graph capture 上下文中测试，环境依赖

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


class TestCudaIsCurrentStreamCapturing(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_returns_bool(self):
        """Verify is_current_stream_capturing returns a boolean."""
        result = torch.npu.is_current_stream_capturing()
        self.assertIsInstance(result, bool)

    def test_npu_not_capturing_by_default(self):
        """Verify stream is not in capture state by default."""
        result = torch.npu.is_current_stream_capturing()
        self.assertFalse(result)


if __name__ == "__main__":
    run_tests()
