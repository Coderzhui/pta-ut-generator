# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.accelerator.set_device_index 接口功能正确性
API 名称：torch.accelerator.set_device_index
API 签名：def set_device_index(device: _device_t, /) -> None

覆盖维度表：
| 覆盖维度         | 说明                                          | 覆盖情况               |
|------------------|-----------------------------------------------|------------------------|
| 基础调用         | 传入合法设备索引不报错                        | 已覆盖                 |
| 参数类型         | int / str / torch.device 三种类型             | 已覆盖                 |
| 传参与不传参     | 单必选参数                                    | 已覆盖                 |
| 状态变更         | 设置后 current_device_index 返回正确值        | 已覆盖                 |
| 异常路径         | 无效设备索引触发 RuntimeError                  | 已覆盖                 |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、设备状态符合预期），
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


class TestAcceleratorSetDeviceIndex(TestCase):
    """Test cases for torch.accelerator.set_device_index."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )
        self._original_device_index = torch.accelerator.current_device_index()

    def tearDown(self):
        # Restore original device index
        try:
            torch.accelerator.set_device_index(self._original_device_index)
        except Exception:
            pass
        super().tearDown()

    def test_set_device_index_int(self):
        """set_device_index with int argument."""
        torch.accelerator.set_device_index(0)
        self.assertEqual(torch.accelerator.current_device_index(), 0)

    def test_set_device_index_str(self):
        """set_device_index with string argument."""
        torch.accelerator.set_device_index(f"{self.device_name}:0")
        self.assertEqual(torch.accelerator.current_device_index(), 0)

    def test_set_device_index_torch_device(self):
        """set_device_index with torch.device argument."""
        dev = torch.device(f"{self.device_name}:0")
        torch.accelerator.set_device_index(dev)
        self.assertEqual(torch.accelerator.current_device_index(), 0)

    def test_set_device_index_returns_none(self):
        """set_device_index returns None."""
        result = torch.accelerator.set_device_index(0)
        self.assertIsNone(result)

    def test_set_device_index_callable(self):
        """set_device_index is callable."""
        self.assertTrue(callable(torch.accelerator.set_device_index))

    def test_set_invalid_index_raises(self):
        """Setting an out-of-range device index should raise."""
        count = torch.accelerator.device_count()
        with self.assertRaises((RuntimeError, AssertionError)):
            torch.accelerator.set_device_index(count + 100)


if __name__ == "__main__":
    run_tests()
