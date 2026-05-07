# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.checkpoint.checkpoint 接口功能正确性
API 名称：torch.utils.checkpoint.checkpoint
API 签名：torch.utils.checkpoint.checkpoint(function, *args, use_reentrant=True, **kwargs)

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | checkpoint 包裹函数不报错           | 已覆盖                  |
| 返回类型         | 返回 Tensor                         | 已覆盖                  |
| NPU 上执行       | NPU tensor 输入                     | 已覆盖                  |
| use_reentrant    | use_reentrant=False                 | 已覆盖                  |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性，不做精度和数值正确性校验。
"""
import torch
import torch_npu  # noqa: F401
from torch.utils.checkpoint import checkpoint

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    import unittest
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


class TestUtilsCheckpointCheckpoint(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")
        self.device = torch.device(self.device_name)

    def test_npu_basic_checkpoint(self):
        """Verify checkpoint wraps a function and returns correct tensor on NPU."""
        def fn(x):
            return x * 2
        x = torch.randn(4, 4, device=self.device, requires_grad=True)
        result = checkpoint(fn, x, use_reentrant=False)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.device.type, self.device_name)

    def test_npu_output_shape(self):
        """Verify checkpoint preserves output shape for scalar-returning function."""
        def fn(x):
            return x.sum()
        x = torch.randn(4, 4, device=self.device, requires_grad=True)
        result = checkpoint(fn, x, use_reentrant=False)
        self.assertEqual(result.shape, torch.Size([]))

    def test_npu_with_grad(self):
        """Verify gradient flows through checkpoint on NPU."""
        def fn(x):
            return x * 3
        x = torch.randn(4, device=self.device, requires_grad=True)
        result = checkpoint(fn, x, use_reentrant=False)
        result.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_cpu_baseline(self):
        """CPU baseline: verify checkpoint works on CPU."""
        def fn(x):
            return x + 1
        x = torch.randn(4, requires_grad=True)
        result = checkpoint(fn, x, use_reentrant=False)
        self.assertIsInstance(result, torch.Tensor)


if __name__ == "__main__":
    run_tests()
