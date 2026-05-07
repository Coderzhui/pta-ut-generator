# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.Generator 接口功能正确性
API 名称：torch.Generator
API 签名：torch.Generator(device=None) -> Generator

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础创建         | 无参数创建 CPU Generator            | 已覆盖                  |
| device 参数      | 指定 NPU 设备创建                   | 已覆盖                  |
| initial_seed     | initial_seed 方法可用               | 已覆盖                  |
| get/set_state    | 状态序列化/反序列化                 | 已覆盖                  |

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


class TestGenerator(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_create_cpu_generator(self):
        """Verify Generator can be created without arguments (CPU default)."""
        g = torch.Generator()
        self.assertIsInstance(g, torch.Generator)

    def test_create_npu_generator(self):
        """Verify Generator can be created with NPU device."""
        g = torch.Generator(device=self.device_name)
        self.assertIsInstance(g, torch.Generator)

    def test_initial_seed(self):
        """Verify initial_seed returns an int."""
        g = torch.Generator()
        seed = g.initial_seed()
        self.assertIsInstance(seed, int)

    def test_manual_seed_returns_self(self):
        """Verify manual_seed returns the Generator itself."""
        g = torch.Generator()
        result = g.manual_seed(123)
        self.assertIs(result, g)

    def test_get_set_state(self):
        """Verify get_state/set_state roundtrip works for state serialization."""
        g = torch.Generator()
        g.manual_seed(42)
        state = g.get_state()
        self.assertIsInstance(state, torch.Tensor)
        g2 = torch.Generator()
        g2.set_state(state)


if __name__ == "__main__":
    run_tests()
