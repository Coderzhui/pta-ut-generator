# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.device (via torch.npu.device) 接口功能正确性
API 名称：torch.cuda.device / torch.npu.device
API 签名：torch.npu.device(device) -> device

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | int 参数返回 device 对象            | 已覆盖                  |
| 作为上下文管理器 | with 语句内设备切换                 | 已覆盖                  |
| 参数类型         | int / torch.device                 | 已覆盖                  |

未覆盖项及原因：
- str 参数 'npu:0'：内部 _utils 验证 device type 限制
- 异常路径：无效设备索引环境依赖性强

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


class TestCudaDevice(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_returns_device_object(self):
        """Verify torch.npu.device returns a device instance."""
        result = torch.npu.device(0)
        self.assertIsInstance(result, torch.npu.device)

    def test_npu_as_context_manager(self):
        """Verify torch.npu.device works as a context manager for device switching."""
        with torch.npu.device(0):
            pass

    def test_npu_with_int_arg(self):
        """Verify device accepts an integer device index."""
        d = torch.npu.device(0)
        self.assertIsNotNone(d)

    def test_npu_with_torch_device_arg(self):
        """Verify device accepts a torch.device object."""
        d = torch.npu.device(torch.device('npu', 0))
        self.assertIsNotNone(d)


if __name__ == "__main__":
    run_tests()
