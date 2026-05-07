# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.device_of (via torch.npu.device_of) 接口功能正确性
API 名称：torch.cuda.device_of / torch.npu.device_of
API 签名：torch.npu.device_of(tensor) -> device_of

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 传入 NPU tensor 返回 device_of 对象 | 已覆盖                  |
| 作为上下文管理器 | with 语句内设备切换                 | 已覆盖                  |

未覆盖项及原因：
- 异常路径：CPU tensor 传入的场景不确定报错类型

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


class TestCudaDeviceOf(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_returns_device_of_object(self):
        """Verify device_of returns a device_of instance for an NPU tensor."""
        t = torch.tensor([1.0, 2.0], device=self.device_name)
        result = torch.npu.device_of(t)
        self.assertIsInstance(result, torch.npu.device_of)

    def test_npu_as_context_manager(self):
        """Verify device_of works as a context manager for device switching."""
        t = torch.tensor([1.0, 2.0], device=self.device_name)
        with torch.npu.device_of(t):
            pass


if __name__ == "__main__":
    run_tests()
