# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.device_memory_used 接口在 NPU 环境的可用性
API 名称：torch.cuda.device_memory_used
API 签名：torch.cuda.device_memory_used(device=None) -> int

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | API 是否存在                        | 未覆盖：torch_npu 无此 API |

未覆盖项及原因：
- torch.npu 命名空间下不存在 device_memory_used，该 API 在 NPU 环境不可用

注意：本测试仅验证 API 存在性。
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


class TestCudaDeviceMemoryUsed(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_no_device_memory_used(self):
        """device_memory_used is not available in torch_npu."""
        self.assertFalse(hasattr(torch.npu, 'device_memory_used'))


if __name__ == "__main__":
    run_tests()
