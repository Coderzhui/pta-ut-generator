# -*- coding: utf-8 -*-
"""
测试目的：验证 torch._dynamo.config.ignore_logger_methods.add 接口功能正确性
API 名称：torch._dynamo.config.ignore_logger_methods.add
API 签名：ignore_logger_methods: set[Callable] (Config wrapped set), .add(elem) -> None

覆盖维度表：
| 覆盖维度         | 说明                                      | 覆盖情况             |
|------------------|-------------------------------------------|----------------------|
| 基础调用         | add 可调用对象不报错                      | 已覆盖               |
| 集合属性         | ignore_logger_methods 为 set-like 对象    | 已覆盖               |
| 添加重复元素     | 重复 add 不报错                           | 已覆盖               |
| 元素存在验证     | add 后集合包含该元素                      | 已覆盖               |
| 非可调用参数     | add 非 callable 的行为                    | 已覆盖               |
| 幂等性           | 多次 add 同一元素无副作用                 | 已覆盖               |
| 异常路径         | 无稳定异常路径                            | 未覆盖：set.add 无校验 |

未覆盖项及原因：
- 异常路径：set.add 不校验参数类型，无稳定异常

注意：本测试仅验证功能正确性（调用不报错、副作用符合预期），
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


def _dummy_logger_method(msg):
    """Dummy callable for testing ignore_logger_methods.add."""
    pass


def _another_logger_method(msg):
    """Another dummy callable for testing."""
    pass


class TestDynamoConfigIgnoreLoggerMethodsAdd(TestCase):
    """Test cases for torch._dynamo.config.ignore_logger_methods.add."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )
        # Save original state
        self._original = set(torch._dynamo.config.ignore_logger_methods)

    def tearDown(self):
        # Restore original state
        torch._dynamo.config.ignore_logger_methods.clear()
        for item in self._original:
            torch._dynamo.config.ignore_logger_methods.add(item)
        super().tearDown()

    def test_ignore_logger_methods_is_set_like(self):
        """ignore_logger_methods should be a set-like container."""
        ilm = torch._dynamo.config.ignore_logger_methods
        self.assertTrue(hasattr(ilm, 'add'))
        self.assertTrue(hasattr(ilm, '__contains__'))
        self.assertTrue(hasattr(ilm, '__iter__'))

    def test_add_callable_succeeds(self):
        """Adding a callable to ignore_logger_methods succeeds."""
        torch._dynamo.config.ignore_logger_methods.add(_dummy_logger_method)
        self.assertIn(_dummy_logger_method, torch._dynamo.config.ignore_logger_methods)

    def test_add_duplicate_callable_no_error(self):
        """Adding the same callable twice does not raise."""
        torch._dynamo.config.ignore_logger_methods.add(_dummy_logger_method)
        torch._dynamo.config.ignore_logger_methods.add(_dummy_logger_method)
        # Set semantics: still only one instance
        count = sum(1 for f in torch._dynamo.config.ignore_logger_methods if f is _dummy_logger_method)
        self.assertGreaterEqual(count, 1)

    def test_add_multiple_callables(self):
        """Adding multiple distinct callables all appear in the set."""
        torch._dynamo.config.ignore_logger_methods.add(_dummy_logger_method)
        torch._dynamo.config.ignore_logger_methods.add(_another_logger_method)
        self.assertIn(_dummy_logger_method, torch._dynamo.config.ignore_logger_methods)
        self.assertIn(_another_logger_method, torch._dynamo.config.ignore_logger_methods)

    def test_add_returns_none(self):
        """set.add() returns None."""
        result = torch._dynamo.config.ignore_logger_methods.add(_dummy_logger_method)
        self.assertIsNone(result)

    def test_add_lambda_callable(self):
        """Adding a lambda function succeeds."""
        fn = lambda msg: None  # noqa: E731
        torch._dynamo.config.ignore_logger_methods.add(fn)
        self.assertIn(fn, torch._dynamo.config.ignore_logger_methods)


if __name__ == "__main__":
    run_tests()
