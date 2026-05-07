# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.qconfig.default_activation_only_qconfig 接口功能正确性
API 名称：torch.ao.quantization.qconfig.default_activation_only_qconfig
API 签名：default_activation_only_qconfig = QConfig(activation=default_fake_quant, weight=torch.nn.Identity)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | QConfig 为模块级单例，activation/weight 均非空               | 已覆盖                                         |
| 枚举选项         | activation 为 FakeQuantize (with_args)，weight 为 nn.Identity | 已覆盖                                         |
| 参数类型         | activation 为 FakeQuantize 子类工厂，weight 为 nn.Identity 类 | 已覆盖                                         |
| 传参与不传参     | QConfig 为预定义单例，不接受额外参数                         | 已覆盖（验证字段不可变性）                     |
| 等价类/边界值    | 多次访问返回同一对象；activation/weight 实例化多次           | 已覆盖                                         |
| 正常传参场景     | 实例化 activation/weight 并执行 forward                      | 已覆盖：2D/4D 张量 forward                    |
| 异常传参场景     | QConfig 为只读单例，无显式异常路径                           | 未覆盖（无稳定异常路径）                       |

未覆盖项及原因：
- 异常传参场景：default_activation_only_qconfig 是预定义的 QConfig namedtuple 单例，无法传参构造，无预期异常
- 数值精度校验：本测试仅验证功能正确性，不做精度和数值正确性校验

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

from torch.ao.quantization.qconfig import QConfig, default_activation_only_qconfig
from torch.ao.quantization.fake_quantize import FakeQuantize


class TestDefaultActivationOnlyQconfig(TestCase):
    """Test cases for torch.ao.quantization.qconfig.default_activation_only_qconfig."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    # ------------------------------------------------------------------
    # 1. Verify it is a QConfig instance
    # ------------------------------------------------------------------
    def test_is_qconfig(self):
        """default_activation_only_qconfig should be a QConfig namedtuple instance."""
        self.assertIsInstance(default_activation_only_qconfig, QConfig)
        self.assertIsInstance(default_activation_only_qconfig, tuple)

    # ------------------------------------------------------------------
    # 2. Verify both activation and weight fields exist and are callable
    # ------------------------------------------------------------------
    def test_has_activation_and_weight(self):
        """QConfig must have activation and weight fields, both callable."""
        self.assertTrue(hasattr(default_activation_only_qconfig, 'activation'))
        self.assertTrue(hasattr(default_activation_only_qconfig, 'weight'))
        self.assertTrue(callable(default_activation_only_qconfig.activation))
        self.assertTrue(callable(default_activation_only_qconfig.weight))

    # ------------------------------------------------------------------
    # 3. Activation instantiation — FakeQuantize
    # ------------------------------------------------------------------
    def test_activation_instantiation(self):
        """activation field should instantiate to FakeQuantize (an nn.Module)."""
        act = default_activation_only_qconfig.activation()
        self.assertIsInstance(act, nn.Module)
        self.assertIsInstance(act, FakeQuantize)

    # ------------------------------------------------------------------
    # 4. Weight instantiation — nn.Identity
    # ------------------------------------------------------------------
    def test_weight_instantiation(self):
        """weight field should instantiate to nn.Identity (an nn.Module)."""
        wt = default_activation_only_qconfig.weight()
        self.assertIsInstance(wt, nn.Module)
        self.assertIsInstance(wt, nn.Identity)

    # ------------------------------------------------------------------
    # 5. Activation forward — shape preserved for 2D input
    # ------------------------------------------------------------------
    def test_activation_forward(self):
        """FakeQuantize forward should preserve input shape and dtype."""
        act = default_activation_only_qconfig.activation()
        x = torch.randn(3, 4)
        out = act(x)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)

    # ------------------------------------------------------------------
    # 6. Weight forward — shape preserved (nn.Identity is a pass-through)
    # ------------------------------------------------------------------
    def test_weight_forward(self):
        """nn.Identity forward should preserve input shape and dtype."""
        wt = default_activation_only_qconfig.weight()
        # 2D tensor
        x_2d = torch.randn(3, 4)
        out_2d = wt(x_2d)
        self.assertEqual(out_2d.shape, x_2d.shape)
        self.assertEqual(out_2d.dtype, x_2d.dtype)

        # 4D tensor (conv weight style)
        x_4d = torch.randn(2, 3, 5, 5)
        out_4d = wt(x_4d)
        self.assertEqual(out_4d.shape, x_4d.shape)
        self.assertEqual(out_4d.dtype, x_4d.dtype)

    # ------------------------------------------------------------------
    # 7. Singleton identity — repeated accesses return the same object
    # ------------------------------------------------------------------
    def test_fields_are_distinct_across_accesses(self):
        """Accessing the module-level qconfig multiple times returns the same singleton."""
        from torch.ao.quantization.qconfig import default_activation_only_qconfig as ref1
        from torch.ao.quantization.qconfig import default_activation_only_qconfig as ref2
        self.assertIs(ref1, ref2)
        import torch.ao.quantization.qconfig as qconfig_mod
        self.assertIs(qconfig_mod.default_activation_only_qconfig, ref1)

    # ------------------------------------------------------------------
    # 8. Activation forward on various tensor shapes
    # ------------------------------------------------------------------
    def test_activation_forward_various_shapes(self):
        """FakeQuantize forward preserves shape for 1D, 3D, 4D tensors."""
        act = default_activation_only_qconfig.activation()
        for shape in [(8,), (2, 3, 4), (1, 5, 6, 7)]:
            with self.subTest(shape=shape):
                x = torch.randn(*shape)
                out = act(x)
                self.assertEqual(out.shape, x.shape)

    # ------------------------------------------------------------------
    # 9. Activation FakeQuantize has observer attribute
    # ------------------------------------------------------------------
    def test_activation_has_observer(self):
        """FakeQuantize instance should have an .activation_post_process attribute that is an nn.Module."""
        act = default_activation_only_qconfig.activation()
        self.assertTrue(hasattr(act, 'activation_post_process'))
        self.assertIsInstance(act.activation_post_process, nn.Module)

    # ------------------------------------------------------------------
    # 10. Weight nn.Identity forward on 1D tensor
    # ------------------------------------------------------------------
    def test_weight_forward_1d(self):
        """nn.Identity forward on 1D tensor should preserve shape and dtype."""
        wt = default_activation_only_qconfig.weight()
        x = torch.randn(10)
        out = wt(x)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)

    # ------------------------------------------------------------------
    # 11. QConfig immutability (namedtuple)
    # ------------------------------------------------------------------
    def test_qconfig_immutability(self):
        """QConfig is a namedtuple; attribute assignment should raise AttributeError."""
        with self.assertRaises(AttributeError):
            default_activation_only_qconfig.activation = FakeQuantize

    # ------------------------------------------------------------------
    # 12. Multiple instantiation produces distinct module instances
    # ------------------------------------------------------------------
    def test_multiple_instantiation_distinct(self):
        """Calling activation() and weight() multiple times yields distinct module instances."""
        act1 = default_activation_only_qconfig.activation()
        act2 = default_activation_only_qconfig.activation()
        self.assertIsNot(act1, act2)

        wt1 = default_activation_only_qconfig.weight()
        wt2 = default_activation_only_qconfig.weight()
        self.assertIsNot(wt1, wt2)


if __name__ == "__main__":
    run_tests()
