# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.set_device (via torch.npu.set_device) 接口功能正确性
API 名称：torch.cuda.set_device / torch.npu.set_device
API 签名：torch.npu.set_device(device) -> None

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | int 参数设置设备返回 None           | 已覆盖                  |
| 参数类型         | int / torch.device                 | 已覆盖                  |
| 切换后验证       | current_device 返回设置的设备       | 已覆盖                  |

未覆盖项及原因：
- 无效设备索引：环境依赖，报错类型不固定

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


class TestCudaSetDevice(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")
        self._orig = torch.npu.current_device()

    def tearDown(self):
        torch.npu.set_device(self._orig)
        super().tearDown()

    def test_npu_returns_none(self):
        """Verify set_device returns None."""
        result = torch.npu.set_device(0)
        self.assertIsNone(result)

    def test_npu_set_device_int(self):
        """Verify set_device with int argument switches active device."""
        torch.npu.set_device(0)
        self.assertEqual(torch.npu.current_device(), 0)

    def test_npu_set_device_torch_device(self):
        """Verify set_device accepts a torch.device object."""
        torch.npu.set_device(torch.device('npu', 0))
        self.assertEqual(torch.npu.current_device(), 0)


if __name__ == "__main__":
    run_tests()
