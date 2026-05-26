# -*- coding: utf-8 -*-
"""
测试目的：验证 torch._dynamo.is_compiling 接口功能正确性
API 名称：torch._dynamo.is_compiling
API 签名：def is_compiling() -> bool

覆盖维度表：
| 覆盖维度         | 说明                                     | 覆盖情况             |
|------------------|------------------------------------------|----------------------|
| 基础调用         | 非编译上下文中返回 False                 | 已覆盖               |
| 返回类型         | 返回 bool                               | 已覆盖               |
| 幂等性           | 多次调用结果一致                         | 已覆盖               |
| 编译上下文内行为 | 在 torch.compile 装饰的函数内返回 True   | 已覆盖               |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、返回类型符合预期），
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


class TestDynamoIsCompiling(TestCase):
    """Test cases for torch._dynamo.is_compiling."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_is_compiling_returns_bool(self):
        """is_compiling should return a bool."""
        result = torch._dynamo.is_compiling()
        self.assertIsInstance(result, bool)

    def test_is_compiling_false_outside_compile(self):
        """is_compiling returns False outside any compile context."""
        result = torch._dynamo.is_compiling()
        self.assertFalse(result)

    def test_is_compiling_idempotent(self):
        """Multiple calls outside compile return consistent results."""
        r1 = torch._dynamo.is_compiling()
        r2 = torch._dynamo.is_compiling()
        self.assertEqual(r1, r2)

    def test_is_compiling_true_inside_compiled_fn(self):
        """is_compiling returns True inside a torch.compile-decorated function."""
        results = []

        @torch.compile(fullgraph=True, backend="eager")
        def inner_fn(x):
            results.append(torch._dynamo.is_compiling())
            return x + x

        x = torch.randn(4, device=self.device_name)
        inner_fn(x)
        self.assertTrue(len(results) > 0)
        self.assertTrue(results[0])

    def test_is_compiling_callable(self):
        """is_compiling is directly callable."""
        self.assertTrue(callable(torch._dynamo.is_compiling))


if __name__ == "__main__":
    run_tests()
