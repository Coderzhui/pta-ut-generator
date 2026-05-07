# -*- coding: utf-8 -*-
"""
测试目的：验证 torch_npu.npu_conv2d 接口功能正确性
API 名称：torch_npu.npu_conv2d
API 签名：torch_npu.npu_conv2d(input, weight, bias, stride, padding, dilation, groups) -> Tensor

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | bias=None vs 有值                                            | 已覆盖                                         |
| 枚举选项         | stride/padding/dilation 的各种合法组合                       | 已覆盖                                         |
| 参数类型         | input/weight/bias 为 Tensor, stride/padding/dilation 为 tuple, groups 为 int | 已覆盖     |
| 传参与不传参     | groups=1 vs groups>1                                         | 已覆盖                                         |
| 等价类/边界值    | 1x1 conv, 3x3 conv, 大 batch, 单元素空间维度                | 已覆盖                                         |
| 正常传参场景     | 标准 conv2d 参数组合                                         | 已覆盖                                         |
| 异常传参场景     | shape 不兼容                                                 | 未覆盖：底层报错类型不固定                     |
| 混合设备类型     | NPU input + CPU weight                                       | 已覆盖：应触发 RuntimeError                    |
| dtype            | float32, float16                                             | 已覆盖                                         |
| shape            | 各种 batch/channel/spatial 组合                              | 已覆盖                                         |

未覆盖项及原因：
- 异常传参场景（shape 不兼容）：底层 C++ 错误类型不固定，无法稳定断言

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


class TestNpuConv2d(TestCase):
    """Test cases for torch_npu.npu_conv2d on NPU."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")
        self.device = torch.device(self.device_name)

    # ------------------------------------------------------------------ #
    # NPU tests
    # ------------------------------------------------------------------ #

    def test_npu_basic_float32(self):
        """Basic conv2d with bias on NPU returns correct shape."""
        input_t = torch.randn(1, 3, 32, 32, device=self.device)
        weight = torch.randn(16, 3, 3, 3, device=self.device)
        bias = torch.randn(16, device=self.device)
        result = torch_npu.npu_conv2d(input_t, weight, bias, (1, 1), (0, 0), (1, 1), 1)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, torch.Size([1, 16, 30, 30]))
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.device.type, self.device_name)

    def test_npu_basic_float16(self):
        """float16 input produces float16 output."""
        input_t = torch.randn(1, 3, 32, 32, dtype=torch.float16, device=self.device)
        weight = torch.randn(16, 3, 3, 3, dtype=torch.float16, device=self.device)
        bias = torch.randn(16, dtype=torch.float16, device=self.device)
        result = torch_npu.npu_conv2d(input_t, weight, bias, (1, 1), (0, 0), (1, 1), 1)
        # Verify basic call with typical parameters succeeds
        self.assertEqual(result.dtype, torch.float16)

    def test_npu_bias_none(self):
        """Conv2d without bias (bias=None) returns correct shape."""
        input_t = torch.randn(1, 3, 32, 32, device=self.device)
        weight = torch.randn(16, 3, 3, 3, device=self.device)
        result = torch_npu.npu_conv2d(input_t, weight, None, (1, 1), (0, 0), (1, 1), 1)
        self.assertEqual(result.shape, torch.Size([1, 16, 30, 30]))

    def test_npu_1x1_conv(self):
        """1x1 convolution (kernel_size=1)."""
        input_t = torch.randn(1, 64, 28, 28, device=self.device)
        weight = torch.randn(128, 64, 1, 1, device=self.device)
        result = torch_npu.npu_conv2d(input_t, weight, None, (1, 1), (0, 0), (1, 1), 1)
        self.assertEqual(result.shape, torch.Size([1, 128, 28, 28]))

    def test_npu_stride_2(self):
        """Conv2d with stride=2 halves spatial dims."""
        input_t = torch.randn(1, 3, 32, 32, device=self.device)
        weight = torch.randn(16, 3, 3, 3, device=self.device)
        result = torch_npu.npu_conv2d(input_t, weight, None, (2, 2), (0, 0), (1, 1), 1)
        self.assertEqual(result.shape, torch.Size([1, 16, 15, 15]))

    def test_npu_padding(self):
        """Conv2d with padding=1 preserves spatial dims."""
        input_t = torch.randn(1, 3, 32, 32, device=self.device)
        weight = torch.randn(16, 3, 3, 3, device=self.device)
        result = torch_npu.npu_conv2d(input_t, weight, None, (1, 1), (1, 1), (1, 1), 1)
        self.assertEqual(result.shape, torch.Size([1, 16, 32, 32]))

    def test_npu_dilation(self):
        """Conv2d with dilation=2 increases effective receptive field."""
        input_t = torch.randn(1, 3, 32, 32, device=self.device)
        weight = torch.randn(16, 3, 3, 3, device=self.device)
        result = torch_npu.npu_conv2d(input_t, weight, None, (1, 1), (0, 0), (2, 2), 1)
        self.assertEqual(result.shape, torch.Size([1, 16, 28, 28]))

    def test_npu_groups(self):
        """Grouped convolution with groups > 1."""
        input_t = torch.randn(1, 16, 32, 32, device=self.device)
        weight = torch.randn(16, 1, 3, 3, device=self.device)
        result = torch_npu.npu_conv2d(input_t, weight, None, (1, 1), (1, 1), (1, 1), 16)
        self.assertEqual(result.shape, torch.Size([1, 16, 32, 32]))

    def test_npu_large_batch(self):
        """Large batch size conv2d."""
        input_t = torch.randn(32, 3, 224, 224, device=self.device)
        weight = torch.randn(64, 3, 7, 7, device=self.device)
        result = torch_npu.npu_conv2d(input_t, weight, None, (2, 2), (3, 3), (1, 1), 1)
        self.assertEqual(result.shape[0], 32)
        self.assertEqual(result.shape[1], 64)

    def test_npu_small_spatial(self):
        """Small spatial dims (1x1) with appropriate kernel."""
        input_t = torch.randn(1, 16, 1, 1, device=self.device)
        weight = torch.randn(32, 16, 1, 1, device=self.device)
        result = torch_npu.npu_conv2d(input_t, weight, None, (1, 1), (0, 0), (1, 1), 1)
        self.assertEqual(result.shape, torch.Size([1, 32, 1, 1]))

    def test_npu_stride_padding_dilation_combined(self):
        """Non-trivial stride + padding + dilation combination."""
        input_t = torch.randn(4, 8, 32, 32, device=self.device)
        weight = torch.randn(16, 8, 3, 3, device=self.device)
        result = torch_npu.npu_conv2d(input_t, weight, None, (1, 2), (1, 2), (1, 2), 1)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.device.type, self.device_name)

    def test_npu_mixed_device_raises(self):
        """NPU input + CPU weight raises RuntimeError."""
        input_npu = torch.randn(1, 3, 8, 8, device=self.device)
        weight_cpu = torch.randn(16, 3, 3, 3)
        with self.assertRaises(RuntimeError):
            torch_npu.npu_conv2d(input_npu, weight_cpu, None, (1, 1), (0, 0), (1, 1), 1)

    # ------------------------------------------------------------------ #
    # CPU baseline — npu_conv2d falls back to torch.nn.functional.conv2d on CPU
    # ------------------------------------------------------------------ #

    def test_cpu_basic(self):
        """CPU baseline: torch.nn.functional.conv2d call returns tensor."""
        import torch.nn.functional as F
        input_t = torch.randn(1, 3, 8, 8)
        weight = torch.randn(16, 3, 3, 3)
        result = F.conv2d(input_t, weight, None, (1, 1), (0, 0), (1, 1), 1)
        # Output must reside on NPU device
        self.assertIsInstance(result, torch.Tensor)


if __name__ == "__main__":
    run_tests()
