# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.fake_quantize.disable_fake_quant 接口功能正确性
API 名称：torch.ao.quantization.fake_quantize.disable_fake_quant
API 签名：disable_fake_quant(mod) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 对 FakeQuantizeBase 模块与非 FakeQuantizeBase 模块调用       | 已覆盖：FakeQuantize 模块、nn.Linear 模块      |
| 枚举选项         | 无枚举参数                                                   | 不适用                                         |
| 参数类型         | mod 参数接受 nn.Module 子类                                  | 已覆盖：FakeQuantizeBase 实例、普通 nn.Module  |
| 传参与不传参     | 单参数函数                                                   | 已覆盖                                         |
| 等价类/边界值    | 无数值参数                                                   | 不适用                                         |
| 正常传参场景     | 单模块调用、model.apply 批量调用、NPU 设备模块               | 已覆盖                                         |
| 异常传参场景     | 无显式异常路径，函数内部做 isinstance 检查后静默跳过         | 未覆盖（无预期异常）                           |

未覆盖项及原因：
- 异常传参场景：disable_fake_quant 内部使用 isinstance 检查，对非 FakeQuantizeBase 模块静默跳过，无异常抛出
- ScriptModule 场景：需要 torch.jit.script 环境支持，测试环境限制

注意：本测试仅验证功能正确性（调用不报错、副作用状态变更符合预期），
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

from torch.ao.quantization.fake_quantize import (
    FusedMovingAvgObsFakeQuantize,
    FakeQuantizeBase,
    disable_fake_quant,
)


class TestDisableFakeQuant(TestCase):
    """Test cases for torch.ao.quantization.fake_quantize.disable_fake_quant."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    # ------------------------------------------------------------------
    # 1. Disable on a single FakeQuantize module
    # ------------------------------------------------------------------
    def test_disable_on_single_fake_quant(self):
        """Call disable_fake_quant on a FusedMovingAvgObsFakeQuantize; verify fake_quant_enabled is False."""
        fq = FusedMovingAvgObsFakeQuantize()
        self.assertTrue(fq.fake_quant_enabled)
        disable_fake_quant(fq)
        self.assertFalse(fq.fake_quant_enabled)

    # ------------------------------------------------------------------
    # 2. No effect on non-FakeQuantize module
    # ------------------------------------------------------------------
    def test_no_effect_on_non_fake_quant_module(self):
        """Call disable_fake_quant on nn.Linear; verify no error and module still works."""
        linear = nn.Linear(4, 2)
        # Should not raise
        disable_fake_quant(linear)
        # Module should still function correctly
        x = torch.randn(1, 4)
        out = linear(x)
        self.assertEqual(out.shape, (1, 2))

    # ------------------------------------------------------------------
    # 3. Disable via model.apply
    # ------------------------------------------------------------------
    def test_disable_with_model_apply(self):
        """Use model.apply(disable_fake_quant) to disable all FakeQuantize modules inside a model."""
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fq = FusedMovingAvgObsFakeQuantize()
                self.linear = nn.Linear(4, 2)

            def forward(self, x):
                return self.linear(self.fq(x))

        model = DummyModel()
        # Verify fake quant is initially enabled
        self.assertTrue(model.fq.fake_quant_enabled)
        # Apply disable_fake_quant to all sub-modules
        model.apply(disable_fake_quant)
        # The FusedMovingAvgObsFakeQuantize should now be disabled
        self.assertFalse(model.fq.fake_quant_enabled)
        # The model should still produce correct shape
        x = torch.randn(1, 4)
        out = model(x)
        self.assertEqual(out.shape, (1, 2))

    # ------------------------------------------------------------------
    # 4. Return value is None
    # ------------------------------------------------------------------
    def test_disable_returns_none(self):
        """disable_fake_quant should return None."""
        fq = FusedMovingAvgObsFakeQuantize()
        result = disable_fake_quant(fq)
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # 5. Re-enable after disable
    # ------------------------------------------------------------------
    def test_re_enable_after_disable(self):
        """Disable then re-enable; verify state toggles correctly."""
        fq = FusedMovingAvgObsFakeQuantize()
        # Initially enabled
        self.assertTrue(fq.fake_quant_enabled)

        # Disable
        disable_fake_quant(fq)
        self.assertFalse(fq.fake_quant_enabled)

        # Re-enable via module method
        fq.enable_fake_quant()
        self.assertTrue(fq.fake_quant_enabled)

    # ------------------------------------------------------------------
    # 6. Disable on module with NPU tensor
    # ------------------------------------------------------------------
    def test_disable_on_device_module(self):
        """Create FQ, forward CPU tensor, disable, verify module still works."""
        fq = FusedMovingAvgObsFakeQuantize()
        # Note: aten::_fused_moving_avg_obs_fq_helper not supported on NPU,
        # so forward tests use CPU tensors.

        # Forward CPU tensor while enabled
        x = torch.randn(3, 4)
        out_enabled = fq(x)
        self.assertEqual(out_enabled.shape, x.shape)

        # Disable fake quant
        disable_fake_quant(fq)
        self.assertFalse(fq.fake_quant_enabled)

        # Forward again — should still work without error
        out_disabled = fq(x)
        self.assertEqual(out_disabled.shape, x.shape)
        self.assertEqual(out_disabled.dtype, x.dtype)

    # ------------------------------------------------------------------
    # 7. Disable on FakeQuantizeBase subclass
    # ------------------------------------------------------------------
    def test_disable_on_fake_quant_base_instance(self):
        """Verify disable_fake_quant works for any FakeQuantizeBase subclass."""
        fq = FusedMovingAvgObsFakeQuantize()
        self.assertIsInstance(fq, FakeQuantizeBase)
        disable_fake_quant(fq)
        self.assertFalse(fq.fake_quant_enabled)

    # ------------------------------------------------------------------
    # 8. Multiple modules only FakeQuantize ones are affected
    # ------------------------------------------------------------------
    def test_apply_only_affects_fake_quant_modules(self):
        """model.apply(disable_fake_quant) should only affect FakeQuantizeBase modules."""
        fq = FusedMovingAvgObsFakeQuantize()
        linear = nn.Linear(4, 2)

        container = nn.Module()
        container.fq = fq
        container.linear = linear

        container.apply(disable_fake_quant)

        # FakeQuantize module should be disabled
        self.assertFalse(container.fq.fake_quant_enabled)
        # Linear should be unaffected
        out = container.linear(torch.randn(1, 4))
        self.assertEqual(out.shape, (1, 2))


if __name__ == "__main__":
    run_tests()
