# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.ns._numeric_suite_fx.loggers_set_save_activations 接口功能正确性
API 名称：torch.ao.ns._numeric_suite_fx.loggers_set_save_activations
API 签名：def loggers_set_save_activations(model: torch.nn.Module, save_activations: bool) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 模型中包含或不包含 OutputLogger 子模块                       | 包含：test_set_save_activations_true/false 等；不包含：test_no_loggers_in_model |
| 枚举选项         | save_activations 参数为 True 或 False                        | True 和 False 均覆盖                           |
| 参数类型         | model 为 nn.Module，save_activations 为 bool                 | 覆盖                                           |
| 传参与不传参     | 必须传两个参数，无默认值                                     | 覆盖正常传参场景                               |
| 等价类/边界值    | 0个/1个/多个 OutputLogger；混合模块模型                      | 覆盖 0个、1个、多个、混合                      |
| 正常传参场景     | 传入有效模型和布尔值                                         | 覆盖                                           |
| 异常传参场景     | 函数本身不做参数校验，仅需确保不报错                         | 未覆盖（无异常抛出场景）                       |

未覆盖项及原因：
- 异常传参场景：该函数内部无参数类型校验，不会主动抛出异常，故不覆盖

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import torch
import torch_npu  # noqa: F401
import torch.nn as nn

from torch.ao.ns._numeric_suite_fx import (
    loggers_set_save_activations,
    OutputLogger,
)

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    from unittest import TestCase

    def run_tests():
        import unittest

        unittest.main()


# Helper: minimal model containing one or more OutputLogger submodules
class _ModelWithOneLogger(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.logger = OutputLogger(
            ref_node_name="ref",
            prev_node_name="prev",
            model_name="m",
            ref_name="r",
            prev_node_target_type="linear",
            ref_node_target_type="linear",
            results_type="output",
            index_within_arg=0,
            index_of_arg=0,
            fqn="logger",
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class _ModelWithMultipleLoggers(nn.Module):
    """Model with OutputLogger instances at different nesting levels."""

    def __init__(self):
        super().__init__()
        self.logger_top = OutputLogger(
            ref_node_name="ref_top",
            prev_node_name="prev_top",
            model_name="m",
            ref_name="r_top",
            prev_node_target_type="relu",
            ref_node_target_type="relu",
            results_type="output",
            index_within_arg=0,
            index_of_arg=0,
            fqn="logger_top",
        )
        self.sub = nn.Sequential(
            nn.Linear(4, 4),
            OutputLogger(
                ref_node_name="ref_sub",
                prev_node_name="prev_sub",
                model_name="m",
                ref_name="r_sub",
                prev_node_target_type="linear",
                ref_node_target_type="linear",
                results_type="output",
                index_within_arg=0,
                index_of_arg=0,
                fqn="sub.logger",
            ),
            nn.ReLU(),
        )
        self.logger_bottom = OutputLogger(
            ref_node_name="ref_bottom",
            prev_node_name="prev_bottom",
            model_name="m",
            ref_name="r_bottom",
            prev_node_target_type="relu",
            ref_node_target_type="relu",
            results_type="output",
            index_within_arg=0,
            index_of_arg=0,
            fqn="logger_bottom",
        )

    def forward(self, x):
        x = self.logger_top(x)
        x = self.sub(x)
        x = self.logger_bottom(x)
        return x


class TestLoggersSetSaveActivations(TestCase):
    """Tests for torch.ao.ns._numeric_suite_fx.loggers_set_save_activations."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    def test_set_save_activations_true(self):
        """Verify that setting save_activations=True updates all OutputLogger instances."""
        model = _ModelWithOneLogger()
        # Ensure starting from False so the change is observable
        model.logger.save_activations = False
        loggers_set_save_activations(model, True)
        self.assertTrue(model.logger.save_activations)

    def test_set_save_activations_false(self):
        """Verify that setting save_activations=False updates all OutputLogger instances."""
        model = _ModelWithOneLogger()
        # Ensure starting from True so the change is observable
        model.logger.save_activations = True
        loggers_set_save_activations(model, False)
        self.assertFalse(model.logger.save_activations)

    def test_no_loggers_in_model(self):
        """Verify the function does not raise on a plain model with no OutputLogger."""
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        # Should complete without error
        result = loggers_set_save_activations(model, True)
        self.assertIsNone(result)

    def test_multiple_loggers(self):
        """Verify all OutputLogger instances at different levels are updated."""
        model = _ModelWithMultipleLoggers()
        # Set initial state to True
        model.logger_top.save_activations = True
        model.logger_bottom.save_activations = True
        # The one inside sub Sequential
        for name, mod in model.named_modules():
            if isinstance(mod, OutputLogger):
                mod.save_activations = True

        loggers_set_save_activations(model, False)

        # Collect all OutputLogger modules and verify each one
        loggers = [
            mod for mod in model.modules() if isinstance(mod, OutputLogger)
        ]
        self.assertGreaterEqual(len(loggers), 3)
        for logger in loggers:
            self.assertFalse(
                logger.save_activations,
                msg=f"Logger {logger} should have save_activations=False",
            )

    def test_mixed_modules(self):
        """Verify only OutputLogger modules are affected; other modules remain unchanged."""
        model = _ModelWithOneLogger()
        loggers_set_save_activations(model, False)
        # Logger should be False
        self.assertFalse(model.logger.save_activations)
        # Non-logger modules should still be their original types (not replaced)
        self.assertIsInstance(model.linear, nn.Linear)
        self.assertIsInstance(model.relu, nn.ReLU)

    def test_return_type(self):
        """Verify the function returns None."""
        model = _ModelWithOneLogger()
        result = loggers_set_save_activations(model, True)
        self.assertIsNone(result)

    def test_idempotent_toggle(self):
        """Verify calling the function twice with the same value is idempotent."""
        model = _ModelWithOneLogger()
        loggers_set_save_activations(model, True)
        self.assertTrue(model.logger.save_activations)
        loggers_set_save_activations(model, True)
        self.assertTrue(model.logger.save_activations)

    def test_toggle_back_and_forth(self):
        """Verify toggling save_activations between True and False works correctly."""
        model = _ModelWithOneLogger()
        loggers_set_save_activations(model, False)
        self.assertFalse(model.logger.save_activations)
        loggers_set_save_activations(model, True)
        self.assertTrue(model.logger.save_activations)
        loggers_set_save_activations(model, False)
        self.assertFalse(model.logger.save_activations)


if __name__ == "__main__":
    run_tests()
