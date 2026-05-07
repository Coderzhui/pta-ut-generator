# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.observer.default_debug_observer 接口功能正确性
API 名称：torch.ao.quantization.observer.default_debug_observer
API 签名：default_debug_observer = RecordingObserver

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | default_debug_observer 即 RecordingObserver 类               | 已覆盖                                         |
| 枚举选项         | 无枚举参数，类引用直接赋值                                   | 已覆盖                                         |
| 参数类型         | 无参数调用创建实例                                           | 已覆盖                                         |
| 传参与不传参     | 无参实例化                                                   | 已覆盖                                         |
| 等价类/边界值    | 多种输入 tensor shape: (3,4), (2,3,4), (5,)                 | 已覆盖                                         |
| 正常传参场景     | 实例化、forward 并验证记录数据                               | 已覆盖                                         |
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

from torch.ao.quantization.observer import default_debug_observer, RecordingObserver


class TestDefaultDebugObserver(TestCase):

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    def test_type(self):
        """Verify default_debug_observer IS the RecordingObserver class."""
        self.assertIs(default_debug_observer, RecordingObserver)

    def test_instantiation(self):
        """Verify that calling default_debug_observer() returns an nn.Module instance."""
        observer = default_debug_observer()
        self.assertIsInstance(observer, nn.Module)

    def test_isinstance_recording_observer(self):
        """Verify the created instance is specifically a RecordingObserver."""
        observer = default_debug_observer()
        self.assertIsInstance(observer, RecordingObserver)

    def test_forward_shape_2d(self):
        """Verify output shape matches input shape for a 2D tensor."""
        observer = default_debug_observer()
        x = torch.randn(3, 4)
        output = observer(x)
        self.assertEqual(output.shape, x.shape)

    def test_forward_shape_3d(self):
        """Verify output shape matches input shape for a 3D tensor."""
        observer = default_debug_observer()
        x = torch.randn(2, 3, 4)
        output = observer(x)
        self.assertEqual(output.shape, x.shape)

    def test_forward_shape_1d(self):
        """Verify output shape matches input shape for a 1D tensor."""
        observer = default_debug_observer()
        x = torch.randn(5)
        output = observer(x)
        self.assertEqual(output.shape, x.shape)

    def test_forward_dtype_preserved(self):
        """Verify output dtype matches input dtype (float32)."""
        observer = default_debug_observer()
        x = torch.randn(3, 4)
        output = observer(x)
        self.assertEqual(output.dtype, x.dtype)

    def test_forward_output_is_tensor(self):
        """Verify the forward output is a torch.Tensor."""
        observer = default_debug_observer()
        x = torch.randn(3, 4)
        output = observer(x)
        self.assertIsInstance(output, torch.Tensor)

    def test_records_tensor_data(self):
        """Verify that RecordingObserver records tensor data after forward pass."""
        observer = default_debug_observer()
        x = torch.randn(3, 4)
        observer(x)
        # get_tensor_value() returns a list of recorded tensors
        tensor_val = observer.get_tensor_value()
        self.assertIsNotNone(tensor_val)
        self.assertIsInstance(tensor_val, list)
        self.assertGreaterEqual(len(tensor_val), 1)
        self.assertEqual(tensor_val[0].shape, x.shape)

    def test_records_multiple_forwards(self):
        """Verify that RecordingObserver records data from multiple forward passes."""
        observer = default_debug_observer()
        x1 = torch.randn(3, 4)
        x2 = torch.randn(2, 3, 4)
        observer(x1)
        observer(x2)
        # After two forward passes, the observer should have recorded both
        all_data = observer.get_tensor_value()
        self.assertIsInstance(all_data, list)
        self.assertGreaterEqual(len(all_data), 2)


if __name__ == "__main__":
    run_tests()
