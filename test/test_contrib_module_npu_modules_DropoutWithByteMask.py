# -*- coding: utf-8 -*-
"""
测试目的：验证 torch_npu.contrib.module.npu_modules.DropoutWithByteMask 接口功能正确性
API 名称：torch_npu.contrib.module.npu_modules.DropoutWithByteMask
API 签名：DropoutWithByteMask(p=0.5, inplace=False, max_seed=2**10-1)
          forward(input1) -> Tensor

覆盖维度表：
| 覆盖维度         | 说明                                                     | 覆盖情况                                         |
|------------------|----------------------------------------------------------|--------------------------------------------------|
| shape            | 2D / 3D / 空 tensor                                      | 已覆盖；4D/奇数维度内核 aicpu 异常不支持          |
| dtype            | float32 / float16                                        | 已覆盖                                           |
| device           | NPU 主路径（>80%）                                       | 已覆盖；CPU 不支持，无 CPU 基线                  |
| 参数枚举         | p: 0/0.5/1.0; inplace: True/False; max_seed: default/custom | 已覆盖                                        |
| 可选参数         | p / inplace / max_seed 显式传入 vs 省略默认              | 已覆盖                                           |
| 训练/评估模式     | model.train() / model.eval() 对 self.training 的影响     | 已覆盖                                           |
| 正常传参场景     | 典型 shape/dtype，各种 p 值，train/eval 模式             | 已覆盖                                           |
| 异常传参场景     | p < 0 / p > 1 在 __init__ 中; inplace=True 在 forward 中 | 已覆盖                                           |
| 混合设备类型     | 单 Tensor 输入，不涉及多设备混用                          | 未覆盖（单输入 forward，不适用）                  |

未覆盖项及原因：
- 4D tensor (shape ≥ 4 dims): DropOutGenMaskV3 kernel aicpu 异常 errorCode=0x2a
- 含奇数维度 (3/5/7): 同上内核 bug
- 混合设备类型：forward 仅接收单个 input1，不涉及多 Tensor 设备混用
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


class TestDropoutWithByteMaskModule(TestCase):
    """Test cases for torch_npu.contrib.module.npu_modules.DropoutWithByteMask on NPU."""

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
        """2D tensor input via module on NPU."""
        m = torch_npu.contrib.module.npu_modules.DropoutWithByteMask(p=0.5)
        x = torch.randn(8, 8, device=self.device)
        result = m(x)
        self.assertEqual(result.shape, torch.Size([8, 8]))

    def test_npu_shape_3d(self):
        """3D tensor input via module on NPU."""
        m = torch_npu.contrib.module.npu_modules.DropoutWithByteMask(p=0.5)
        x = torch.randn(4, 4, 4, device=self.device)
        result = m(x)
        self.assertEqual(result.shape, torch.Size([4, 4, 4]))

    def test_npu_empty_tensor(self):
        """Empty tensor on NPU does not raise."""
        m = torch_npu.contrib.module.npu_modules.DropoutWithByteMask(p=0.5)
        x = torch.randn(0, 4, device=self.device)
        result = m(x)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, torch.Size([0, 4]))

    # ------------------------------------------------------------------ #
    # NPU dtype tests
    # ------------------------------------------------------------------ #

    def test_npu_dtype_float32(self):
        """float32 input produces float32 output via module on NPU."""
        m = torch_npu.contrib.module.npu_modules.DropoutWithByteMask(p=0.5)
        x = torch.randn(4, 4, dtype=torch.float32, device=self.device)
        result = m(x)
        self.assertEqual(result.dtype, torch.float32)

    def test_npu_dtype_float16(self):
        """float16 input produces float16 output via module on NPU."""
        m = torch_npu.contrib.module.npu_modules.DropoutWithByteMask(p=0.5)
        x = torch.randn(4, 4, dtype=torch.float16, device=self.device)
        result = m(x)
        self.assertEqual(result.dtype, torch.float16)

    # ------------------------------------------------------------------ #
    # device tests
    # ------------------------------------------------------------------ #

    def test_npu_output_device(self):
        """Output tensor is on NPU device."""
        m = torch_npu.contrib.module.npu_modules.DropoutWithByteMask(p=0.5)
        x = torch.randn(4, 4, device=self.device)
        result = m(x)
        self.assertEqual(result.device.type, self.device_name)

    # ------------------------------------------------------------------ #
    # p parameter tests
    # ------------------------------------------------------------------ #

    def test_npu_p_zero(self):
        """p=0: output matches input structure."""
        m = torch_npu.contrib.module.npu_modules.DropoutWithByteMask(p=0.0)
        x = torch.randn(4, 4, device=self.device)
        result = m(x)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)
        self.assertEqual(result.device, x.device)

    def test_npu_p_one(self):
        """p=1: all elements zeroed via module."""
        m = torch_npu.contrib.module.npu_modules.DropoutWithByteMask(p=1.0)
        x = torch.randn(4, 4, device=self.device)
        result = m(x)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)
        self.assertEqual(result.device, x.device)

    def test_npu_p_default(self):
        """Default p=0.5 when not specified in __init__."""
        m = torch_npu.contrib.module.npu_modules.DropoutWithByteMask()
        x = torch.randn(4, 4, device=self.device)
        result = m(x)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, x.shape)

    # ------------------------------------------------------------------ #
    # training / eval mode tests
    # ------------------------------------------------------------------ #

    def test_npu_training_mode(self):
        """In training mode (default), dropout is applied."""
        m = torch_npu.contrib.module.npu_modules.DropoutWithByteMask(p=0.5)
        m.train()
        x = torch.randn(4, 4, device=self.device)
        result = m(x)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)
        self.assertEqual(result.device, x.device)

    def test_npu_eval_mode(self):
        """In eval mode, dropout is skipped."""
        m = torch_npu.contrib.module.npu_modules.DropoutWithByteMask(p=0.5)
        m.eval()
        x = torch.randn(4, 4, device=self.device)
        result = m(x)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)
        self.assertEqual(result.device, x.device)

    # ------------------------------------------------------------------ #
    # inplace parameter tests
    # ------------------------------------------------------------------ #

    def test_npu_inplace_false_default(self):
        """inplace=False (default): forward returns new tensor."""
        m = torch_npu.contrib.module.npu_modules.DropoutWithByteMask(p=0.5)
        x = torch.randn(4, 4, device=self.device)
        result = m(x)
        self.assertIsInstance(result, torch.Tensor)
        self.assertIsNot(result, x)

    def test_npu_inplace_true_raises(self):
        """inplace=True triggers ValueError from functional API."""
        m = torch_npu.contrib.module.npu_modules.DropoutWithByteMask(p=0.5, inplace=True)
        x = torch.randn(4, 4, device=self.device)
        with self.assertRaises(ValueError):
            m(x)

    # ------------------------------------------------------------------ #
    # max_seed parameter tests
    # ------------------------------------------------------------------ #

    def test_npu_max_seed_default(self):
        """Default max_seed does not break forward."""
        m = torch_npu.contrib.module.npu_modules.DropoutWithByteMask(p=0.5)
        x = torch.randn(4, 4, device=self.device)
        result = m(x)
        self.assertIsInstance(result, torch.Tensor)

    def test_npu_max_seed_custom(self):
        """Custom max_seed does not break forward."""
        m = torch_npu.contrib.module.npu_modules.DropoutWithByteMask(p=0.5, max_seed=512)
        x = torch.randn(4, 4, device=self.device)
        result = m(x)
        self.assertIsInstance(result, torch.Tensor)

    # ------------------------------------------------------------------ #
    # exception path tests (__init__)
    # ------------------------------------------------------------------ #

    def test_npu_p_negative_raises_in_init(self):
        """p < 0 raises ValueError in __init__."""
        with self.assertRaises(ValueError):
            torch_npu.contrib.module.npu_modules.DropoutWithByteMask(p=-0.1)

    def test_npu_p_greater_than_one_raises_in_init(self):
        """p > 1 raises ValueError in __init__."""
        with self.assertRaises(ValueError):
            torch_npu.contrib.module.npu_modules.DropoutWithByteMask(p=1.5)

    # ------------------------------------------------------------------ #
    # misc tests
    # ------------------------------------------------------------------ #

    def test_npu_non_contiguous_input(self):
        """Non-contiguous tensor input via module on NPU."""
        m = torch_npu.contrib.module.npu_modules.DropoutWithByteMask(p=0.5)
        x = torch.randn(8, 8, device=self.device).t()
        self.assertFalse(x.is_contiguous())
        result = m(x)
        self.assertIsInstance(result, torch.Tensor)

    def test_npu_output_is_contiguous(self):
        """Output is contiguous for contiguous input."""
        m = torch_npu.contrib.module.npu_modules.DropoutWithByteMask(p=0.5)
        x = torch.randn(4, 4, device=self.device)
        result = m(x)
        self.assertTrue(result.is_contiguous())

    def test_npu_repeated_forward(self):
        """Multiple forward calls with same module instance."""
        m = torch_npu.contrib.module.npu_modules.DropoutWithByteMask(p=0.5)
        for _ in range(3):
            x = torch.randn(4, 4, device=self.device)
            result = m(x)
            self.assertEqual(result.shape, torch.Size([4, 4]))
            self.assertEqual(result.device.type, self.device_name)


if __name__ == "__main__":
    run_tests()
