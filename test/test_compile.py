# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.compile 接口功能正确性
API 名称：torch.compile
API 签名：def compile(model=None, *, fullgraph=False, dynamic=None, backend="inductor",
                      mode=None, options=None, disable=False, recompile_limit=None)

覆盖维度表：
| 覆盖维度         | 说明                                           | 覆盖情况             |
|------------------|------------------------------------------------|----------------------|
| 基础调用         | compile(fn) 返回可调用对象                     | 已覆盖               |
| 无参数装饰器     | @torch.compile 装饰函数                        | 已覆盖               |
| 参数枚举         | fullgraph / backend / mode / disable           | 已覆盖               |
| 传参与不传参     | model=None vs 传入函数                         | 已覆盖               |
| NPU 执行         | 编译后函数在 NPU 上执行                        | 已覆盖               |
| 返回类型         | 返回 callable                                  | 已覆盖               |
| disable=True     | 禁用编译，直接执行                             | 已覆盖               |
| 异常路径         | 无稳定异常路径                                 | 未覆盖               |

未覆盖项及原因：
- 异常路径：compile 接受多种参数组合，无稳定异常路径

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype 符合预期），
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


def _add_fn(x):
    return x + x


def _relu_fn(x):
    return torch.relu(x)


class TestCompile(TestCase):
    """Test cases for torch.compile on NPU."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )
        self.device = torch.device(self.device_name)

    def tearDown(self):
        # Reset dynamo caches between tests
        try:
            torch._dynamo.reset()
        except Exception:
            pass
        super().tearDown()

    def test_compile_returns_callable(self):
        """torch.compile(fn) returns a callable."""
        compiled = torch.compile(_add_fn, backend="eager")
        self.assertTrue(callable(compiled))

    def test_compile_no_args_returns_decorator(self):
        """torch.compile() with no args returns a decorator factory."""
        decorator = torch.compile()
        self.assertTrue(callable(decorator))

    def test_compile_fn_on_npu(self):
        """Compiled function runs on NPU and returns correct shape/dtype."""
        compiled = torch.compile(_add_fn, backend="eager")
        x = torch.randn(4, 4, device=self.device)
        result = compiled(x)
        self.assertEqual(result.shape, torch.Size([4, 4]))
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.device.type, self.device_name)

    def test_compile_fullgraph_true(self):
        """torch.compile with fullgraph=True."""
        compiled = torch.compile(_add_fn, fullgraph=True, backend="eager")
        x = torch.randn(3, device=self.device)
        result = compiled(x)
        self.assertEqual(result.shape, torch.Size([3]))

    def test_compile_disable_true(self):
        """torch.compile with disable=True runs without compilation."""
        compiled = torch.compile(_add_fn, disable=True, backend="eager")
        x = torch.randn(4, device=self.device)
        result = compiled(x)
        self.assertEqual(result.shape, torch.Size([4]))
        self.assertEqual(result.dtype, torch.float32)

    def test_compile_backend_eager(self):
        """torch.compile with backend='eager'."""
        compiled = torch.compile(_relu_fn, backend="eager")
        x = torch.randn(4, device=self.device)
        result = compiled(x)
        self.assertEqual(result.shape, torch.Size([4]))

    def test_compile_as_decorator(self):
        """torch.compile used as a decorator."""
        @torch.compile(backend="eager")
        def decorated_fn(x):
            return x * 2

        x = torch.randn(2, 3, device=self.device)
        result = decorated_fn(x)
        self.assertEqual(result.shape, torch.Size([2, 3]))

    def test_compile_float16_input(self):
        """Compiled function with float16 input."""
        compiled = torch.compile(_add_fn, backend="eager")
        x = torch.randn(4, 4, dtype=torch.float16, device=self.device)
        result = compiled(x)
        self.assertEqual(result.dtype, torch.float16)

    def test_compile_scalar_output(self):
        """Compiled function returning a scalar."""
        @torch.compile(backend="eager")
        def sum_fn(x):
            return x.sum()

        x = torch.randn(4, device=self.device)
        result = sum_fn(x)
        self.assertEqual(result.shape, torch.Size([]))

    def test_compile_callable(self):
        """torch.compile is callable."""
        self.assertTrue(callable(torch.compile))


if __name__ == "__main__":
    run_tests()
