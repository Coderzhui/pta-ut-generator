# -*- coding: utf-8 -*-
"""
测试目的：验证 tensor.numpy 接口功能正确性
API 名称：tensor.numpy
API 签名：tensor.numpy(force=False) -> numpy.ndarray

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | CPU tensor 转 numpy                 | 已覆盖                  |
| 返回类型         | 返回 numpy.ndarray                  | 已覆盖                  |
| shape 保留       | 输出 shape 与输入一致               | 已覆盖                  |
| dtype 保留       | 输出 dtype 与输入一致               | 已覆盖                  |
| NPU tensor       | NPU tensor 先 .cpu() 再 .numpy()    | 已覆盖                  |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性，不做精度和数值正确性校验。
"""
import numpy as np
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


class TestTensorNumpy(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_cpu_tensor_numpy(self):
        """Verify .numpy() converts CPU tensor to ndarray."""
        t = torch.randn(4, 8)
        result = t.numpy()
        self.assertIsInstance(result, np.ndarray)

    def test_shape_preserved(self):
        """Verify output shape matches input tensor shape."""
        t = torch.randn(3, 4, 5)
        result = t.numpy()
        self.assertEqual(result.shape, (3, 4, 5))

    def test_dtype_float32(self):
        """Verify float32 dtype is preserved in conversion."""
        t = torch.randn(4, 4, dtype=torch.float32)
        result = t.numpy()
        self.assertEqual(result.dtype, np.float32)

    def test_dtype_int64(self):
        """Verify int64 dtype is preserved in conversion."""
        t = torch.randint(0, 10, (4,), dtype=torch.int64)
        result = t.numpy()
        self.assertEqual(result.dtype, np.int64)

    def test_npu_tensor_via_cpu(self):
        """Verify NPU tensor can be converted via .cpu().numpy()."""
        t = torch.randn(4, 4, device=self.device_name)
        result = t.cpu().numpy()
        self.assertIsInstance(result, np.ndarray)

    def test_force_param(self):
        """Verify force=True parameter works without error."""
        t = torch.randn(4, 4)
        result = t.numpy(force=True)
        self.assertIsInstance(result, np.ndarray)


if __name__ == "__main__":
    run_tests()
