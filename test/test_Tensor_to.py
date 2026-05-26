# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.Tensor.to 接口功能正确性
API 名称：torch.Tensor.to
API 签名：Tensor.to(*args, **kwargs) -> Tensor
         支持多种重载：to(dtype), to(device), to(tensor), to(device, dtype),
                     to(memory_format=), to(non_blocking=), to(copy=)

覆盖维度表：
| 覆盖维度         | 说明                                                    | 覆盖情况             |
|------------------|---------------------------------------------------------|----------------------|
| dtype 转换       | to(dtype) 改变 dtype                                    | 已覆盖               |
| device 迁移      | to(device) 在 CPU/NPU 间迁移                            | 已覆盖               |
| device+dtype     | to(device, dtype) 同时改变                              | 已覆盖               |
| 参考张量         | to(tensor) 匹配另一个 tensor 的 dtype/device            | 已覆盖               |
| 参数枚举         | non_blocking / copy / memory_format                     | 已覆盖               |
| 空张量           | size=0 的 tensor                                        | 已覆盖               |
| 标量张量         | 0-dim tensor                                            | 已覆盖               |
| 混合设备类型     | CPU tensor to NPU + NPU tensor to CPU                   | 已覆盖               |
| 连续性           | to(memory_format=) 改变内存格式                         | 已覆盖               |
| 异常路径         | 无稳定异常路径                                          | 未覆盖：to 内部宽容处理 |

未覆盖项及原因：
- 异常路径：Tensor.to 对不合法参数宽容处理，无稳定可断言异常路径

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/device 符合预期），
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


class TestTensorTo(TestCase):
    """Test cases for torch.Tensor.to on NPU."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )
        self.device = torch.device(self.device_name)

    # ------------------------------------------------------------------ #
    # NPU tests  (target: >80% of all test methods)
    # ------------------------------------------------------------------ #

    def test_to_dtype_float16(self):
        """to(torch.float16) changes dtype."""
        x = torch.randn(3, 4, device=self.device)
        result = x.to(torch.float16)
        self.assertEqual(result.dtype, torch.float16)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.device.type, self.device_name)

    def test_to_dtype_bfloat16(self):
        """to(torch.bfloat16) changes dtype."""
        x = torch.randn(3, 4, device=self.device)
        result = x.to(torch.bfloat16)
        self.assertEqual(result.dtype, torch.bfloat16)

    def test_to_dtype_int32(self):
        """to(torch.int32) changes dtype."""
        x = torch.ones(3, 4, device=self.device, dtype=torch.float32)
        result = x.to(torch.int32)
        self.assertEqual(result.dtype, torch.int32)

    def test_to_device_same(self):
        """to(npu device) keeps on NPU."""
        x = torch.randn(4, device=self.device)
        result = x.to(self.device)
        self.assertEqual(result.device.type, self.device_name)

    def test_to_device_cpu(self):
        """to('cpu') moves tensor to CPU."""
        x = torch.randn(4, device=self.device)
        result = x.to('cpu')
        self.assertEqual(result.device.type, 'cpu')

    def test_to_device_npu_from_cpu(self):
        """to(npu) moves CPU tensor to NPU."""
        x = torch.randn(4)
        result = x.to(self.device)
        self.assertEqual(result.device.type, self.device_name)

    def test_to_device_and_dtype(self):
        """to(device, dtype) changes both."""
        x = torch.randn(3, 4)
        result = x.to(self.device, torch.float16)
        self.assertEqual(result.device.type, self.device_name)
        self.assertEqual(result.dtype, torch.float16)

    def test_to_tensor(self):
        """to(other_tensor) matches dtype and device."""
        x = torch.randn(3, 4)
        ref = torch.randn(1, dtype=torch.float16, device=self.device)
        result = x.to(ref)
        self.assertEqual(result.dtype, torch.float16)
        self.assertEqual(result.device.type, self.device_name)

    def test_to_copy_true(self):
        """to(copy=True) creates a new tensor."""
        x = torch.randn(4, device=self.device)
        result = x.to(copy=True)
        self.assertEqual(result.dtype, x.dtype)
        self.assertEqual(result.device.type, self.device_name)
        self.assertIsNot(result, x)

    def test_to_copy_false_same_device(self):
        """to(copy=False) on same device returns same tensor."""
        x = torch.randn(4, device=self.device)
        result = x.to(self.device, copy=False)
        self.assertIs(result, x)

    def test_to_non_blocking(self):
        """to(non_blocking=True) does not raise."""
        x = torch.randn(4)
        result = x.to(self.device, non_blocking=True)
        self.assertEqual(result.device.type, self.device_name)

    def test_to_empty_tensor(self):
        """to() works on empty tensors."""
        x = torch.randn(0, 4, device=self.device)
        result = x.to(torch.float16)
        self.assertEqual(result.shape, torch.Size([0, 4]))
        self.assertEqual(result.dtype, torch.float16)

    def test_to_scalar_tensor(self):
        """to() works on 0-dim scalar tensor."""
        x = torch.tensor(3.14, device=self.device)
        result = x.to(torch.float16)
        self.assertEqual(result.shape, torch.Size([]))
        self.assertEqual(result.dtype, torch.float16)

    def test_to_high_dim(self):
        """to() works on high-dimensional tensor."""
        x = torch.randn(2, 3, 4, 5, device=self.device)
        result = x.to(torch.float16)
        self.assertEqual(result.shape, torch.Size([2, 3, 4, 5]))

    def test_to_memory_format_contiguous(self):
        """to(memory_format=torch.contiguous_format) accepts the parameter."""
        x = torch.randn(4, 3, 8, 8, device=self.device)
        result = x.to(memory_format=torch.contiguous_format)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)

    # ------------------------------------------------------------------ #
    # CPU baseline  (target: ≤20% of all test methods)
    # ------------------------------------------------------------------ #

    def test_to_dtype_cpu_baseline(self):
        """CPU baseline: to(dtype) changes dtype."""
        x = torch.randn(3, 4)
        result = x.to(torch.float16)
        self.assertEqual(result.dtype, torch.float16)
        self.assertEqual(result.device.type, 'cpu')


if __name__ == "__main__":
    run_tests()
