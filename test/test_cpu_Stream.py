# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cpu.Stream 接口功能正确性
API 名称：torch.cpu.Stream
API 签名：torch.cpu.Stream()

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础创建         | 创建 Stream 不报错                  | 已覆盖                  |
| 上下文管理器     | torch.cpu.stream(stream) 可用       | 已覆盖                  |

未覆盖项及原因：
- query/synchronize：torch.cpu.Stream 无这些方法（不同于 NPU Stream）

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


class TestCpuStream(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_cpu_create_stream(self):
        """Verify torch.cpu.Stream() creates a valid stream instance."""
        s = torch.cpu.Stream()
        self.assertIsInstance(s, torch.cpu.Stream)

    def test_cpu_stream_context_manager(self):
        """Verify torch.cpu.stream works as a context manager."""
        s = torch.cpu.Stream()
        with torch.cpu.stream(s):
            pass


if __name__ == "__main__":
    run_tests()
