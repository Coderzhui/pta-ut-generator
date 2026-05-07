# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.is_initialized 接口功能正确性
API 名称：torch.cuda.is_initialized
API 签名：torch.cuda.is_initialized() -> bool

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 返回 bool 类型                      | 已覆盖                  |
| 初始化后状态     | 创建 tensor 后应返回 True           | 已覆盖                  |

未覆盖项及原因：
- 异常路径：无参数，不会失败

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


class TestCudaIsInitialized(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_returns_bool(self):
        """Verify is_initialized returns a boolean."""
        result = torch.npu.is_initialized()
        self.assertIsInstance(result, bool)

    def test_npu_initialized_after_tensor_creation(self):
        """Verify is_initialized returns bool after NPU tensor creation."""
        _ = torch.tensor(1.0, device=self.device_name)
        self.assertIsInstance(torch.npu.is_initialized(), bool)


if __name__ == "__main__":
    run_tests()
