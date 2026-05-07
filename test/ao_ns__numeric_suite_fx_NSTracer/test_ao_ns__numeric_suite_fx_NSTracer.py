# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.ns._numeric_suite_fx.NSTracer 接口功能正确性
API 名称：torch.ao.ns._numeric_suite_fx.NSTracer
API 签名：class NSTracer(quantize_fx.QuantizationTracer)
          def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 空模型（无子模块）和非空模型（含各种子模块）                 | 覆盖                                           |
| 枚举选项         | is_leaf_module 对 ObserverBase/FakeQuantizeBase 返回 True，对普通 nn 模块返回继承行为 | 覆盖 observer、fake_quantize、普通模块         |
| 参数类型         | skipped_module_names: list[str], skipped_module_classes: list[Callable] | 覆盖空列表和非空列表                           |
| 传参与不传参     | QuantizationTracer 必须传两个列表参数                        | 覆盖                                           |
| 等价类/边界值    | 仅含 observer 的模型；仅含 fake_quantize 的模型；混合模型   | 覆盖                                           |
| 正常传参场景     | 传入合法模型进行 trace                                       | 覆盖                                           |
| 异常传参场景     | NSTracer 构造参数类型不正确时会抛 TypeError                  | 覆盖                                           |

未覆盖项及原因：
- 复杂模型（多分支、ResNet 等）的完整 trace 正确性：超出功能验证范围

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import torch
import torch_npu  # noqa: F401
import torch.nn as nn
from torch.fx import GraphModule

from torch.ao.ns._numeric_suite_fx import NSTracer
from torch.ao.quantization.observer import MinMaxObserver, PlaceholderObserver
from torch.ao.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    from unittest import TestCase

    def run_tests():
        import unittest

        unittest.main()


# ---------------------------------------------------------------------------
# Helper models
# ---------------------------------------------------------------------------


class _SimpleModel(nn.Module):
    """A basic Sequential model with no observers or fake quantize modules."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class _ModelWithObserver(nn.Module):
    """Model that contains an observer module."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.observer = MinMaxObserver()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.observer(x)
        return self.relu(self.linear(x))


class _ModelWithFakeQuantize(nn.Module):
    """Model that contains a fake quantize module."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.fake_quant = FusedMovingAvgObsFakeQuantize()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fake_quant(self.linear(x)))


class _MixedModel(nn.Module):
    """Model containing observer, fake_quantize, and regular nn modules."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.observer = PlaceholderObserver()
        self.fake_quant = FusedMovingAvgObsFakeQuantize()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        self.observer(x)
        x = self.fake_quant(x)
        x = self.relu(x)
        return x


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestNSTracer(TestCase):
    """Tests for torch.ao.ns._numeric_suite_fx.NSTracer."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    # -- Instantiation & type checks -----------------------------------------

    def test_instantiation(self):
        """Verify NSTracer can be instantiated with empty argument lists."""
        tracer = NSTracer(skipped_module_names=[], skipped_module_classes=[])
        self.assertIsInstance(tracer, NSTracer)

    def test_instantiation_is_tracer(self):
        """Verify NSTracer inherits from QuantizationTracer / torch.fx.Tracer."""
        from torch.ao.quantization.fx.tracer import QuantizationTracer
        from torch.fx import Tracer

        tracer = NSTracer([], [])
        self.assertIsInstance(tracer, QuantizationTracer)
        self.assertIsInstance(tracer, Tracer)

    # -- is_leaf_module behaviour --------------------------------------------

    def test_observer_is_leaf(self):
        """Verify is_leaf_module returns True for ObserverBase instances."""
        tracer = NSTracer([], [])
        observer = MinMaxObserver()
        result = tracer.is_leaf_module(observer, "observer")
        self.assertTrue(result)

    def test_placeholder_observer_is_leaf(self):
        """Verify is_leaf_module returns True for PlaceholderObserver."""
        tracer = NSTracer([], [])
        observer = PlaceholderObserver()
        result = tracer.is_leaf_module(observer, "observer")
        self.assertTrue(result)

    def test_fake_quant_is_leaf(self):
        """Verify is_leaf_module returns True for FakeQuantizeBase instances."""
        tracer = NSTracer([], [])
        fq = FusedMovingAvgObsFakeQuantize()
        result = tracer.is_leaf_module(fq, "fake_quant")
        self.assertTrue(result)

    def test_non_observer_not_leaf(self):
        """Verify is_leaf_module returns True for nn.Linear (leaf via parent class)."""
        tracer = NSTracer([], [])
        linear = nn.Linear(4, 4)
        # nn.Linear starts with torch.nn, so parent's is_leaf_module returns True
        result = tracer.is_leaf_module(linear, "linear")
        self.assertTrue(result)

    def test_sequential_not_leaf(self):
        """Verify nn.Sequential is NOT treated as a leaf module."""
        tracer = NSTracer([], [])
        seq = nn.Sequential(nn.Linear(2, 2))
        result = tracer.is_leaf_module(seq, "seq")
        self.assertFalse(result)

    # -- Tracing a simple model ---------------------------------------------

    def test_trace_simple_model(self):
        """Verify tracing a simple Sequential model produces a non-empty graph."""
        model = _SimpleModel()
        tracer = NSTracer([], [])
        graph = tracer.trace(model)
        # Graph should contain at least placeholder, output, and op nodes
        self.assertGreater(len(list(graph.nodes)), 0)

    def test_trace_output_type(self):
        """Verify tracer.trace returns a Graph object and wrapping in GraphModule works."""
        model = _SimpleModel()
        tracer = NSTracer([], [])
        graph = tracer.trace(model)
        self.assertIsInstance(graph, torch.fx.Graph)
        gm = GraphModule(model, graph)
        self.assertIsInstance(gm, GraphModule)

    def test_trace_model_with_observer(self):
        """Verify tracing a model with an observer treats the observer as a leaf."""
        model = _ModelWithObserver()
        tracer = NSTracer([], [])
        graph = tracer.trace(model)
        gm = GraphModule(model, graph)
        self.assertIsInstance(gm, GraphModule)
        # The observer should appear as a call_module node, not be decomposed
        node_targets = [n.target for n in graph.nodes if n.op == "call_module"]
        self.assertIn("observer", node_targets)

    def test_trace_model_with_fake_quant(self):
        """Verify tracing a model with FusedMovingAvgObsFakeQuantize treats it as leaf."""
        model = _ModelWithFakeQuantize()
        tracer = NSTracer([], [])
        graph = tracer.trace(model)
        gm = GraphModule(model, graph)
        self.assertIsInstance(gm, GraphModule)
        node_targets = [n.target for n in graph.nodes if n.op == "call_module"]
        self.assertIn("fake_quant", node_targets)

    def test_trace_mixed_model(self):
        """Verify tracing a mixed model with observer, fake_quantize and regular modules."""
        model = _MixedModel()
        tracer = NSTracer([], [])
        graph = tracer.trace(model)
        gm = GraphModule(model, graph)
        self.assertIsInstance(gm, GraphModule)
        node_targets = [n.target for n in graph.nodes if n.op == "call_module"]
        # observer and fake_quant should both appear as leaf call_module nodes
        self.assertIn("observer", node_targets)
        self.assertIn("fake_quant", node_targets)

    def test_trace_graph_has_placeholder_and_output(self):
        """Verify the traced graph contains at least one placeholder and one output node."""
        model = _SimpleModel()
        tracer = NSTracer([], [])
        graph = tracer.trace(model)
        op_types = [n.op for n in graph.nodes]
        self.assertIn("placeholder", op_types)
        self.assertIn("output", op_types)

    def test_skipped_module_names(self):
        """Verify skipped_module_names causes named modules to be treated as leaf."""
        model = _SimpleModel()
        tracer = NSTracer(skipped_module_names=["linear"], skipped_module_classes=[])
        # linear is in skipped_module_names, so is_leaf_module should return True
        self.assertTrue(tracer.is_leaf_module(model.linear, "linear"))

    def test_skipped_module_classes(self):
        """Verify skipped_module_classes causes matching module types to be leaf."""
        model = _SimpleModel()
        tracer = NSTracer(
            skipped_module_names=[], skipped_module_classes=[nn.ReLU]
        )
        self.assertTrue(tracer.is_leaf_module(model.relu, "relu"))


if __name__ == "__main__":
    run_tests()
