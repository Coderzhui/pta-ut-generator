# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.observer.default_placeholder_observer 接口功能正确性
API 名称：torch.ao.quantization.observer.default_placeholder_observer
API 签名：default_placeholder_observer = PlaceholderObserver

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | default_placeholder_observer 即 PlaceholderObserver 类       | 已覆盖                                         |
| 枚举选项         | 无枚举参数，类引用直接赋值                                   | 已覆盖                                         |
| 参数类型         | 无参数调用创建实例                                           | 已覆盖                                         |
| 传参与不传参     | 无参实例化                                                   | 已覆盖                                         |
| 等价类/边界值    | 多种输入 tensor shape: (3,4), (2,3,4), (5,)                 | 已覆盖                                         |
| 正常传参场景     | 实例化、forward 多种 shape tensor                            | 已覆盖                                         |
| 异常传参场景     | 不适用（无参 API）                                           | 未覆盖                                         |

未覆盖项及原因：
- 异常传参场景：该 API 为类引用直接赋值，无异常场景。

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import torch
import torch_npu  # noqa: F401
import torch.nn as nn

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        import unittest
        unittest.main(argv=sys.argv)

from torch.ao.quantization.observer import default_placeholder_observer, PlaceholderObserver


class TestDefaultPlaceholderObserver(TestCase):

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    def test_type(self):
        """Verify default_placeholder_observer IS the PlaceholderObserver class."""
        self.assertIs(default_placeholder_observer, PlaceholderObserver)

    def test_instantiation(self):
        """Verify that calling default_placeholder_observer() returns an nn.Module instance."""
        observer = default_placeholder_observer()
        self.assertIsInstance(observer, nn.Module)

    def test_isinstance_placeholder_observer(self):
        """Verify the created instance is specifically a PlaceholderObserver."""
        observer = default_placeholder_observer()
        self.assertIsInstance(observer, PlaceholderObserver)

    def test_default_dtype(self):
        """Verify the default dtype is torch.float32."""
        observer = default_placeholder_observer()
        self.assertEqual(observer.dtype, torch.float32)

    def test_forward_shape_2d(self):
        """Verify output shape matches input shape for a 2D tensor."""
        observer = default_placeholder_observer()
        x = torch.randn(3, 4)
        output = observer(x)
        self.assertEqual(output.shape, x.shape)

    def test_forward_shape_3d(self):
        """Verify output shape matches input shape for a 3D tensor."""
        observer = default_placeholder_observer()
        x = torch.randn(2, 3, 4)
        output = observer(x)
        self.assertEqual(output.shape, x.shape)

    def test_forward_shape_1d(self):
        """Verify output shape matches input shape for a 1D tensor."""
        observer = default_placeholder_observer()
        x = torch.randn(5)
        output = observer(x)
        self.assertEqual(output.shape, x.shape)

    def test_forward_dtype_preserved(self):
        """Verify output dtype matches input dtype (float32)."""
        observer = default_placeholder_observer()
        x = torch.randn(3, 4)
        output = observer(x)
        self.assertEqual(output.dtype, x.dtype)

    def test_forward_output_is_tensor(self):
        """Verify the forward output is a torch.Tensor."""
        observer = default_placeholder_observer()
        x = torch.randn(3, 4)
        output = observer(x)
        self.assertIsInstance(output, torch.Tensor)


if __name__ == "__main__":
    run_tests()
