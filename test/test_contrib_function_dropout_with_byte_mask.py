# -*- coding: utf-8 -*-
"""
测试目的：验证 torch_npu.contrib.function.dropout_with_byte_mask 接口功能正确性
API 名称：torch_npu.contrib.function.dropout_with_byte_mask
API 签名：dropout_with_byte_mask(input1, p=0.5, training=True, inplace=False) -> Tensor

覆盖维度表：
| 覆盖维度         | 说明                                                     | 覆盖情况                                         |
|------------------|----------------------------------------------------------|--------------------------------------------------|
| shape            | 2D / 3D / 空 tensor                                      | 已覆盖；4D/奇数维度内核 aicpu 异常不支持          |
| dtype            | float32 / float16                                        | 已覆盖                                           |
| device           | NPU 主路径（>80%）                                       | 已覆盖；CPU 不支持，无 CPU 基线                  |
| 参数枚举         | training: True/False; inplace: True/False                | 已覆盖                                           |
| 可选参数         | p / training / inplace 显式传入 vs 省略默认              | 已覆盖                                           |
| p 值边界         | 0 / 0.5 / 1.0                                            | 已覆盖                                           |
| 正常传参场景     | 典型 shape/dtype 组合，各 p 值，training 开关            | 已覆盖                                           |
| 异常传参场景     | p < 0 / p > 1 / inplace=True                             | 已覆盖                                           |
| 混合设备类型     | 单 Tensor 输入，不涉及多设备混用                          | 未覆盖（单输入 API，不适用）                     |

未覆盖项及原因：
- 4D tensor (shape ≥ 4 dims): DropOutGenMaskV3 kernel aicpu 异常 errorCode=0x2a
- 含奇数维度 (3/5/7): 同上内核 bug
- 混合设备类型：API 仅接收单个 input1 Tensor，不涉及多 Tensor 设备混用
- CPU 基线：API 文档声明"仅支持 NPU 设备"

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/device 符合预期），
     不做精度和数值正确性校验。
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


class TestDropoutWithByteMaskFunction(TestCase):
    """Test cases for torch_npu.contrib.function.dropout_with_byte_mask on NPU."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")
        self.device = torch.device(self.device_name)

    # ------------------------------------------------------------------ #
    # NPU shape tests
    # ------------------------------------------------------------------ #

    def test_npu_shape_2d(self):
        """2D tensor input on NPU."""
        x = torch.randn(8, 8, device=self.device)
        result = torch_npu.contrib.function.dropout_with_byte_mask(x)
        self.assertEqual(result.shape, torch.Size([8, 8]))

    def test_npu_shape_3d(self):
        """3D tensor input on NPU."""
        x = torch.randn(4, 4, 4, device=self.device)
        result = torch_npu.contrib.function.dropout_with_byte_mask(x)
        self.assertEqual(result.shape, torch.Size([4, 4, 4]))

    def test_npu_empty_tensor(self):
        """Empty tensor (size-0 dim) on NPU does not raise."""
        x = torch.randn(0, 4, device=self.device)
        result = torch_npu.contrib.function.dropout_with_byte_mask(x)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, torch.Size([0, 4]))

    # ------------------------------------------------------------------ #
    # NPU dtype tests
    # ------------------------------------------------------------------ #

    def test_npu_dtype_float32(self):
        """float32 input produces float32 output on NPU."""
        x = torch.randn(4, 4, dtype=torch.float32, device=self.device)
        result = torch_npu.contrib.function.dropout_with_byte_mask(x)
        self.assertEqual(result.dtype, torch.float32)

    def test_npu_dtype_float16(self):
        """float16 input produces float16 output on NPU."""
        x = torch.randn(4, 4, dtype=torch.float16, device=self.device)
        result = torch_npu.contrib.function.dropout_with_byte_mask(x)
        self.assertEqual(result.dtype, torch.float16)

    # ------------------------------------------------------------------ #
    # device tests
    # ------------------------------------------------------------------ #

    def test_npu_output_device(self):
        """Output tensor is on NPU device."""
        x = torch.randn(4, 4, device=self.device)
        result = torch_npu.contrib.function.dropout_with_byte_mask(x)
        self.assertEqual(result.device.type, self.device_name)

    # ------------------------------------------------------------------ #
    # p parameter tests
    # ------------------------------------------------------------------ #

    def test_npu_p_zero(self):
        """p=0: output matches input structure."""
        x = torch.randn(4, 4, device=self.device)
        result = torch_npu.contrib.function.dropout_with_byte_mask(x, p=0.0)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)
        self.assertEqual(result.device, x.device)

    def test_npu_p_one(self):
        """p=1: all elements zeroed."""
        x = torch.randn(4, 4, device=self.device)
        result = torch_npu.contrib.function.dropout_with_byte_mask(x, p=1.0)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)
        self.assertEqual(result.device, x.device)

    def test_npu_p_default(self):
        """Default p=0.5 and default training=True when not specified."""
        x = torch.randn(4, 4, device=self.device)
        result = torch_npu.contrib.function.dropout_with_byte_mask(x)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)
        self.assertEqual(result.device, x.device)

    # ------------------------------------------------------------------ #
    # training parameter tests
    # ------------------------------------------------------------------ #

    def test_npu_training_true(self):
        """training=True: dropout is applied."""
        x = torch.randn(4, 4, device=self.device)
        result = torch_npu.contrib.function.dropout_with_byte_mask(x, training=True)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)
        self.assertEqual(result.device, x.device)

    def test_npu_training_false(self):
        """training=False: dropout is skipped."""
        x = torch.randn(4, 4, device=self.device)
        result = torch_npu.contrib.function.dropout_with_byte_mask(x, training=False)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)
        self.assertEqual(result.device, x.device)

    # ------------------------------------------------------------------ #
    # inplace parameter tests
    # ------------------------------------------------------------------ #

    def test_npu_inplace_false(self):
        """inplace=False: returns new tensor."""
        x = torch.randn(4, 4, device=self.device)
        result = torch_npu.contrib.function.dropout_with_byte_mask(x, inplace=False)
        self.assertIsInstance(result, torch.Tensor)
        self.assertIsNot(result, x)

    def test_npu_inplace_true_raises(self):
        """inplace=True raises ValueError (not supported)."""
        x = torch.randn(4, 4, device=self.device)
        with self.assertRaises(ValueError):
            torch_npu.contrib.function.dropout_with_byte_mask(x, inplace=True)

    # ------------------------------------------------------------------ #
    # exception path tests
    # ------------------------------------------------------------------ #

    def test_npu_p_negative_raises(self):
        """p < 0 raises ValueError."""
        x = torch.randn(4, 4, device=self.device)
        with self.assertRaises(ValueError):
            torch_npu.contrib.function.dropout_with_byte_mask(x, p=-0.1)

    def test_npu_p_greater_than_one_raises(self):
        """p > 1 raises ValueError."""
        x = torch.randn(4, 4, device=self.device)
        with self.assertRaises(ValueError):
            torch_npu.contrib.function.dropout_with_byte_mask(x, p=1.5)

    # ------------------------------------------------------------------ #
    # misc tests
    # ------------------------------------------------------------------ #

    def test_npu_non_contiguous_input(self):
        """Non-contiguous tensor input on NPU."""
        x = torch.randn(8, 8, device=self.device).t()
        self.assertFalse(x.is_contiguous())
        result = torch_npu.contrib.function.dropout_with_byte_mask(x)
        self.assertIsInstance(result, torch.Tensor)

    def test_npu_output_is_contiguous(self):
        """Output is contiguous for contiguous input."""
        x = torch.randn(4, 4, device=self.device)
        result = torch_npu.contrib.function.dropout_with_byte_mask(x)
        self.assertTrue(result.is_contiguous())


if __name__ == "__main__":
    run_tests()
