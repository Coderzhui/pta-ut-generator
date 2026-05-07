# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cpu.stream 接口功能正确性
API 名称：torch.cpu.stream
API 签名：torch.cpu.stream(stream)

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 作为上下文管理器不报错              | 已覆盖                  |

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


class TestCpuStreamFn(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_cpu_stream_context_manager(self):
        """Verify torch.cpu.stream works as a context manager."""
        s = torch.cpu.Stream()
        with torch.cpu.stream(s):
            pass


if __name__ == "__main__":
    run_tests()
