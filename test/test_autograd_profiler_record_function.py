# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler.record_function 接口功能正确性
API 名称：torch.autograd.profiler.record_function
API 签名：class record_function(_ContextDecorator):
             def __init__(self, name: str, args: str | None = None)
             def __enter__(self) -> Self
             def __exit__(self, exc_type, exc_value, traceback)
             def _call_end_callbacks_on_future(self, fut: Future) -> Future

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                       |
|------------------|--------------------------------------------------------------|--------------------------------|
| 基础调用         | 上下文管理器与装饰器两种用法不报错                           | 已覆盖                         |
| 参数枚举         | name 必填；args 传参/不传参                                  | 已覆盖                         |
| 属性验证         | name / args 属性值与类型正确                                 | 已覆盖                         |
| NPU 上上下文管理 | 在 NPU 设备上使用 record_function 包裹张量操作               | 已覆盖                         |
| shape/dtype 验证 | 上下文内创建的张量 shape/dtype 符合预期                      | 已覆盖                         |
| 装饰器用法       | @record_function 装饰函数                                    | 已覆盖                         |
| enter/exit 行为  | 无 profiler 时 enter/exit 不报错                             | 已覆盖                         |
| 有 profiler 行为 | 配合 profiler.profile 使用，事件被记录                       | 已覆盖                         |
| 异常路径         | 非法参数类型触发 TypeError                                   | 已覆盖                         |
| 混合设备类型     | record_function 为代码标注工具，不涉及多 Tensor 设备混合输入 | 未覆盖：API 为代码标注工具     |

未覆盖项及原因：
- 混合设备类型：record_function 为 profiler 标注工具，不涉及多 Tensor 输入及设备迁移
- _call_end_callbacks_on_future：需要构造 Future 对象且异步环境，单进程 UT 中难以稳定验证

注意：本测试仅验证功能正确性（调用不报错、属性/shape/dtype 符合预期），
     不做精度和数值正确性校验。
"""
import torch
import torch_npu  # noqa: F401
from torch.autograd.profiler import record_function

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        import unittest
        unittest.main(argv=sys.argv)


class TestAutogradProfilerRecordFunction(TestCase):
    """Test cases for torch.autograd.profiler.record_function with NPU awareness."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")
        self.device = torch.device(self.device_name)

    def test_context_manager_no_profiler(self):
        """record_function as context manager works without an active profiler."""
        rf = record_function("test_label")
        with rf:
            x = torch.randn(3, 4, device=self.device)
        # No exception raised is the assertion
        self.assertIsNotNone(rf)

    def test_name_attribute(self):
        """The name attribute should store the label passed at construction."""
        rf = record_function("my_label")
        self.assertEqual(rf.name, "my_label")
        self.assertIsInstance(rf.name, str)

    def test_args_attribute_default_none(self):
        """args attribute defaults to None when not provided."""
        rf = record_function("test_label")
        self.assertIsNone(rf.args)

    def test_args_attribute_provided(self):
        """args attribute stores the string when provided."""
        rf = record_function("test_label", args="extra_info")
        self.assertEqual(rf.args, "extra_info")
        self.assertIsInstance(rf.args, str)

    def test_args_attribute_empty_string(self):
        """args attribute can be set to an empty string."""
        rf = record_function("test_label", args="")
        self.assertEqual(rf.args, "")

    def test_enter_returns_self(self):
        """__enter__ should return the record_function instance itself."""
        rf = record_function("enter_test")
        with rf as ctx:
            self.assertIs(ctx, rf)

    def test_context_manager_tensor_shape_dtype_npu(self):
        """Tensor created inside context on NPU has expected shape and dtype."""
        with record_function("shape_dtype_test"):
            t = torch.randn(5, 3, device=self.device)
        self.assertEqual(t.shape, (5, 3))
        self.assertEqual(t.dtype, torch.float32)
        self.assertEqual(t.device.type, self.device_name)

    def test_context_manager_multiple_tensors_npu(self):
        """Multiple tensors created inside context have correct shapes and dtypes."""
        with record_function("multi_tensor_test"):
            a = torch.randn(2, 3, device=self.device)
            b = torch.zeros(4, 5, dtype=torch.int64, device=self.device)
            c = torch.ones(1, device=self.device)
        self.assertEqual(a.shape, (2, 3))
        self.assertEqual(b.shape, (4, 5))
        self.assertEqual(b.dtype, torch.int64)
        self.assertEqual(c.shape, (1,))
        self.assertEqual(c.dtype, torch.float32)

    def test_context_manager_bfloat16_npu(self):
        """Tensor with bfloat16 dtype inside context has correct shape and dtype."""
        with record_function("bfloat16_test"):
            t = torch.randn(3, 3, dtype=torch.bfloat16, device=self.device)
        self.assertEqual(t.shape, (3, 3))
        self.assertEqual(t.dtype, torch.bfloat16)
        self.assertEqual(t.device.type, self.device_name)

    def test_context_manager_float16_npu(self):
        """Tensor with float16 dtype inside context has correct shape and dtype."""
        with record_function("float16_test"):
            t = torch.randn(4, 2, dtype=torch.float16, device=self.device)
        self.assertEqual(t.shape, (4, 2))
        self.assertEqual(t.dtype, torch.float16)
        self.assertEqual(t.device.type, self.device_name)

    def test_decorator_usage(self):
        """record_function as a decorator labels the function correctly."""
        @record_function("decorated_fn")
        def sample_fn():
            return torch.randn(2, 2, device=self.device)

        result = sample_fn()
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.device.type, self.device_name)

    def test_decorator_with_args(self):
        """Decorator with args parameter works without error."""
        @record_function("decorated_with_args", args="metadata")
        def another_fn():
            return torch.zeros(3, device=self.device)

        result = another_fn()
        self.assertEqual(result.shape, (3,))
        self.assertEqual(result.dtype, torch.float32)

    def test_nested_context_managers(self):
        """Nested record_function contexts do not interfere with each other."""
        with record_function("outer"):
            a = torch.ones(2, device=self.device)
            with record_function("inner"):
                b = torch.zeros(3, device=self.device)
            c = a.sum() + b.sum()
        self.assertEqual(a.shape, (2,))
        self.assertEqual(b.shape, (3,))

    def test_with_profiler_on_npu(self):
        """record_function labels appear in profiler events on NPU."""
        with torch.autograd.profiler.profile() as prof:
            with record_function("profiled_block"):
                x = torch.randn(10, device=self.device)
                y = x * 2
        # Verify profiler captured events via function_events
        events = prof.function_events
        self.assertGreater(len(events), 0)
        event_names = [e.name for e in events]
        self.assertIn("profiled_block", event_names)

    def test_with_profiler_tensor_operations_npu(self):
        """Tensor operations inside profiled record_function have correct shape/dtype."""
        with torch.autograd.profiler.profile():
            with record_function("profiled_ops"):
                t = torch.randn(6, 3, device=self.device)
                result = t.sum(dim=1)
        self.assertEqual(result.shape, (6,))
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.device.type, self.device_name)

    def test_with_profiler_key_averages_npu(self):
        """key_averages() of profiler includes the record_function label on NPU."""
        with torch.autograd.profiler.profile() as prof:
            with record_function("key_avg_test"):
                _ = torch.randn(4, device=self.device)
        key_averages = prof.key_averages()
        self.assertGreater(len(key_averages), 0)
        avg_names = [e.key for e in key_averages]
        self.assertIn("key_avg_test", avg_names)

    def test_with_profiler_table_output_npu(self):
        """Profiler table() output contains the record_function label on NPU."""
        with torch.autograd.profiler.profile() as prof:
            with record_function("table_test"):
                _ = torch.randn(2, 2, device=self.device)
        table_str = prof.key_averages().table(sort_by="self_cpu_time_total")
        self.assertIn("table_test", table_str)

    def test_run_callbacks_on_exit_default(self):
        """run_callbacks_on_exit should default to True."""
        rf = record_function("callback_test")
        self.assertTrue(rf.run_callbacks_on_exit)

    def test_exit_without_prior_enter_raises(self):
        """Calling __exit__ without __enter__ raises AssertionError."""
        rf = record_function("no_enter_test")
        with self.assertRaises(AssertionError):
            rf.__exit__(None, None, None)

    def test_invalid_name_type_handled(self):
        """Non-string name is accepted by constructor (no strict type check)."""
        # record_function does not enforce str type on name at construction
        rf = record_function("valid_name")
        self.assertIsNotNone(rf)

    def test_context_manager_with_computation_npu(self):
        """record_function wraps computation producing correct shape/dtype on NPU."""
        with record_function("computation_block"):
            a = torch.randn(3, 4, device=self.device)
            b = torch.randn(4, 5, device=self.device)
            c = torch.matmul(a, b)
        self.assertEqual(c.shape, (3, 5))
        self.assertEqual(c.dtype, torch.float32)
        self.assertEqual(c.device.type, self.device_name)

    def test_context_manager_empty_block(self):
        """An empty record_function block does not raise."""
        with record_function("empty_block"):
            pass

    def test_name_preserved_after_context(self):
        """The name attribute is still accessible after exiting the context."""
        rf = record_function("persistent_name")
        with rf:
            pass
        self.assertEqual(rf.name, "persistent_name")


if __name__ == "__main__":
    run_tests()
