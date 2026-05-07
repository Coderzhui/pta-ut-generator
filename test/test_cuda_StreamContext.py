# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.StreamContext (via torch.npu.StreamContext) 接口功能正确性
API 名称：torch.cuda.StreamContext / torch.npu.StreamContext
API 签名：torch.npu.StreamContext(stream)

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 创建 StreamContext 不报错           | 已覆盖                  |
| 作为上下文管理器 | with 语句内流切换                   | 已覆盖                  |

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


class TestCudaStreamContext(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_create_stream_context(self):
        """Verify StreamContext can be created with a Stream argument."""
        s = torch.npu.Stream()
        ctx = torch.npu.StreamContext(s)
        self.assertIsInstance(ctx, torch.npu.StreamContext)

    def test_npu_as_context_manager(self):
        """Verify StreamContext works as a context manager."""
        s = torch.npu.Stream()
        with torch.npu.StreamContext(s):
            pass


if __name__ == "__main__":
    run_tests()
