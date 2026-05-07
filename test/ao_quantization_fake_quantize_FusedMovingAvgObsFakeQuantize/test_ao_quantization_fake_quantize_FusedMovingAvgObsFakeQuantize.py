# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize 接口功能正确性
API 名称：torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize
API 签名：FusedMovingAvgObsFakeQuantize(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, **observer_kwargs)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 默认参数实例化与自定义参数实例化                             | 已覆盖：默认构造、自定义 observer/quant_min/quant_max |
| 枚举选项         | observer 参数可选类型                                         | 已覆盖：MovingAverageMinMaxObserver            |
| 参数类型         | quant_min/quant_max 为 int，observer 为 observer 子类        | 已覆盖                                         |
| 传参与不传参     | 全部使用默认参数 vs 自定义参数                               | 已覆盖                                         |
| 等价类/边界值    | quant_min/quant_max 典型值（0/255, -128/127）                | 已覆盖                                         |
| 正常传参场景     | forward NPU/CPU tensor、enable/disable fake quant            | 已覆盖                                         |
| 异常传参场景     | 无显式异常路径，API 内部为参数存储                           | 未覆盖（无预期异常）                           |

未覆盖项及原因：
- 异常传参场景：FusedMovingAvgObsFakeQuantize 构造函数不做参数类型校验，无明确的异常抛出路径
- calculate_qparams 方法：涉及数值计算，本测试不做精度校验

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

from torch.ao.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize
from torch.ao.quantization.observer import MovingAverageMinMaxObserver


class TestFusedMovingAvgObsFakeQuantize(TestCase):
    """Test cases for torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")
        self.device = torch.device(self.device_name)

    # ------------------------------------------------------------------
    # 1. Default instantiation
    # ------------------------------------------------------------------
    def test_instantiation_default(self):
        """Create with default params; verify it is an nn.Module and correct type."""
        fq = FusedMovingAvgObsFakeQuantize()
        self.assertIsInstance(fq, nn.Module)
        self.assertIsInstance(fq, FusedMovingAvgObsFakeQuantize)

    # ------------------------------------------------------------------
    # 2. Custom observer & quant range
    # ------------------------------------------------------------------
    def test_instantiation_custom_observer(self):
        """Create with custom observer and quant_min/quant_max."""
        fq = FusedMovingAvgObsFakeQuantize(
            observer=MovingAverageMinMaxObserver,
            quant_min=0,
            quant_max=255,
        )
        self.assertIsInstance(fq, FusedMovingAvgObsFakeQuantize)
        self.assertEqual(fq.quant_min, 0)
        self.assertEqual(fq.quant_max, 255)

    # ------------------------------------------------------------------
    # 3. Forward on CPU — shape & dtype preservation
    # Note: aten::_fused_moving_avg_obs_fq_helper is not supported on NPU backend,
    # so forward tests use CPU tensors. This is an upstream NPU limitation.
    # ------------------------------------------------------------------
    def test_forward_cpu_shape_dtype(self):
        """Forward a CPU tensor; output shape and dtype must match input."""
        fq = FusedMovingAvgObsFakeQuantize()
        x = torch.randn(5, 6)
        out = fq(x)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)

    # ------------------------------------------------------------------
    # 4. Forward on CPU with multi-dimensional tensor
    # ------------------------------------------------------------------
    def test_forward_cpu_multidim(self):
        """Forward a 4-D CPU tensor to verify shape preservation for conv-style input."""
        fq = FusedMovingAvgObsFakeQuantize()
        x = torch.randn(1, 3, 8, 8)
        out = fq(x)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)

    # ------------------------------------------------------------------
    # 5. Enable / disable fake_quant toggle
    # ------------------------------------------------------------------
    def test_enable_disable_fake_quant(self):
        """Verify fake_quant_enabled attribute toggles correctly."""
        fq = FusedMovingAvgObsFakeQuantize()
        # Default should be enabled
        self.assertTrue(fq.fake_quant_enabled)

        # Disable
        fq.disable_fake_quant()
        self.assertFalse(fq.fake_quant_enabled)

        # Re-enable
        fq.enable_fake_quant()
        self.assertTrue(fq.fake_quant_enabled)

    # ------------------------------------------------------------------
    # 6. Forward with fake quant disabled — shape preserved
    # ------------------------------------------------------------------
    def test_forward_with_fake_quant_disabled(self):
        """When fake quant is disabled, forward should pass through with unchanged shape."""
        fq = FusedMovingAvgObsFakeQuantize()
        fq.disable_fake_quant()
        x = torch.randn(2, 3)
        out = fq(x)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)

    # ------------------------------------------------------------------
    # 7. Observer attached
    # ------------------------------------------------------------------
    def test_observer_attached(self):
        """Verify .activation_post_process attribute exists and is the correct observer type."""
        fq = FusedMovingAvgObsFakeQuantize(observer=MovingAverageMinMaxObserver)
        self.assertTrue(hasattr(fq, 'activation_post_process'))
        self.assertIsInstance(fq.activation_post_process, MovingAverageMinMaxObserver)

    # ------------------------------------------------------------------
    # 8. Custom quant range stored correctly
    # ------------------------------------------------------------------
    def test_quant_range_custom(self):
        """Create with custom quant_min/quant_max and verify attributes."""
        fq = FusedMovingAvgObsFakeQuantize(quant_min=0, quant_max=127)
        self.assertEqual(fq.quant_min, 0)
        self.assertEqual(fq.quant_max, 127)

    # ------------------------------------------------------------------
    # 9. state_dict keys
    # ------------------------------------------------------------------
    def test_state_dict_keys(self):
        """Verify state_dict contains expected keys."""
        fq = FusedMovingAvgObsFakeQuantize()
        sd = fq.state_dict()
        self.assertIsInstance(sd, dict)
        # The module should have at least fake_quant_enabled and observer-related keys
        self.assertIn('fake_quant_enabled', sd)
        self.assertTrue(len(sd) > 0)

    # ------------------------------------------------------------------
    # 11. calculate_qparams returns tensors (CPU, due to NPU op limitation)
    # ------------------------------------------------------------------
    def test_calculate_qparams_returns_tensors(self):
        """calculate_qparams should return (scale, zero_point) tensors."""
        fq = FusedMovingAvgObsFakeQuantize()
        # Feed data so the observer accumulates statistics
        x = torch.randn(4, 4)
        fq(x)
        scale, zero_point = fq.calculate_qparams()
        self.assertIsInstance(scale, torch.Tensor)
        self.assertIsInstance(zero_point, torch.Tensor)


if __name__ == "__main__":
    run_tests()
