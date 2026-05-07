# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.memory_snapshot (via torch.npu.memory_snapshot) 接口功能正确性
API 名称：torch.cuda.memory_snapshot / torch.npu.memory_snapshot
API 签名：torch.npu.memory_snapshot() -> list

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 返回 list 类型                      | 已覆盖                  |

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


class TestCudaMemorySnapshot(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_returns_list(self):
        """Verify memory_snapshot returns a list."""
        result = torch.npu.memory_snapshot()
        self.assertIsInstance(result, list)


if __name__ == "__main__":
    run_tests()
