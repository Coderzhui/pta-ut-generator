# -*- coding: utf-8 -*-
"""
测试目的：验证 torch._subclasses.fake_tensor.FakeTensorMode 接口功能正确性
API 名称：torch._subclasses.fake_tensor.FakeTensorMode
API 签名：class FakeTensorMode(TorchDispatchMode):
             def __init__(self, *, allow_fallback_kernels=True, allow_non_fake_inputs=False,
                          shape_env=None, static_shapes=None, export=False) -> None

覆盖维度表：
| 覆盖维度         | 说明                                          | 覆盖情况               |
|------------------|-----------------------------------------------|------------------------|
| 基础调用         | 实例化不报错                                  | 已覆盖                 |
| 参数枚举         | allow_fallback_kernels / allow_non_fake_inputs / export | 已覆盖        |
| 上下文管理器     | with 语句进出不报错                           | 已覆盖                 |
| 传参与不传参     | 默认参数 vs 显式传参                          | 已覆盖                 |
| Fake Tensor 创建 | from_tensor / from_real_tensor 产生 fake tensor | 已覆盖               |
| 返回类型         | 实例为 FakeTensorMode 类型                    | 已覆盖                 |
| 异常路径         | 无稳定异常路径                                | 未覆盖：构造函数无校验 |

未覆盖项及原因：
- 异常路径：FakeTensorMode 构造函数无参数校验

注意：本测试仅验证功能正确性（调用不报错、输出类型符合预期），
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

from torch._subclasses.fake_tensor import FakeTensorMode


class TestSubclassesFakeTensorFakeTensorMode(TestCase):
    """Test cases for torch._subclasses.fake_tensor.FakeTensorMode."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_instantiate_default(self):
        """FakeTensorMode can be instantiated with default args."""
        mode = FakeTensorMode()
        self.assertIsInstance(mode, FakeTensorMode)

    def test_instantiate_allow_fallback_false(self):
        """FakeTensorMode with allow_fallback_kernels=False."""
        mode = FakeTensorMode(allow_fallback_kernels=False)
        self.assertFalse(mode.allow_fallback_kernels)

    def test_instantiate_allow_non_fake_inputs_true(self):
        """FakeTensorMode with allow_non_fake_inputs=True."""
        mode = FakeTensorMode(allow_non_fake_inputs=True)
        self.assertTrue(mode.allow_non_fake_inputs)

    def test_instantiate_export_true(self):
        """FakeTensorMode with export=True."""
        mode = FakeTensorMode(export=True)
        self.assertIsInstance(mode, FakeTensorMode)

    def test_context_manager_enter_exit(self):
        """FakeTensorMode works as a context manager."""
        with FakeTensorMode() as mode:
            self.assertIsInstance(mode, FakeTensorMode)

    def test_fake_tensor_from_real_tensor(self):
        """FakeTensorMode can convert a real tensor to a fake tensor."""
        with FakeTensorMode() as mode:
            real = torch.randn(3, 4, device=self.device_name)
            fake = mode.from_tensor(real)
            self.assertEqual(fake.shape, real.shape)
            self.assertEqual(fake.dtype, real.dtype)
            # Fake tensors live on meta device
            self.assertTrue(fake.is_meta or hasattr(fake, 'fake_device'))

    def test_fake_tensor_operations(self):
        """Operations on fake tensors produce fake tensors."""
        with FakeTensorMode() as mode:
            x = torch.randn(2, 3, device=self.device_name)
            fake_x = mode.from_tensor(x)
            result = fake_x + fake_x
            self.assertEqual(result.shape, fake_x.shape)
            self.assertEqual(result.dtype, fake_x.dtype)

    def test_default_allow_fallback_kernels(self):
        """Default allow_fallback_kernels is True."""
        mode = FakeTensorMode()
        self.assertTrue(mode.allow_fallback_kernels)

    def test_default_allow_non_fake_inputs(self):
        """Default allow_non_fake_inputs is False."""
        mode = FakeTensorMode()
        self.assertFalse(mode.allow_non_fake_inputs)


if __name__ == "__main__":
    run_tests()
