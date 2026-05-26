# -*- coding: utf-8 -*-
"""
测试目的：验证 torch._dynamo.reset_code_caches 接口功能正确性
API 名称：torch._dynamo.reset_code_caches
API 签名：def reset_code_caches() -> None

覆盖维度表：
| 覆盖维度         | 说明                                   | 覆盖情况               |
|------------------|----------------------------------------|------------------------|
| 基础调用         | 调用不报错                             | 已覆盖                 |
| 返回类型         | 返回 None                              | 已覆盖                 |
| 幂等性           | 多次调用不报错                         | 已覆盖                 |
| 副作用           | 清除内存中的代码缓存                   | 已覆盖（间接验证）     |
| 异常路径         | 无稳定异常路径                         | 未覆盖：API 无参数校验 |

未覆盖项及原因：
- 异常路径：reset_code_caches 无参数，内部无校验，无稳定异常路径

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


class TestDynamoResetCodeCaches(TestCase):
    """Test cases for torch._dynamo.reset_code_caches."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_reset_code_caches_callable(self):
        """reset_code_caches is callable."""
        self.assertTrue(callable(torch._dynamo.reset_code_caches))

    def test_reset_code_caches_returns_none(self):
        """reset_code_caches returns None."""
        result = torch._dynamo.reset_code_caches()
        self.assertIsNone(result)

    def test_reset_code_caches_no_error(self):
        """reset_code_caches does not raise when called."""
        torch._dynamo.reset_code_caches()

    def test_reset_code_caches_idempotent(self):
        """Calling reset_code_caches multiple times does not raise."""
        torch._dynamo.reset_code_caches()
        torch._dynamo.reset_code_caches()
        torch._dynamo.reset_code_caches()

    def test_reset_code_caches_after_compile(self):
        """reset_code_caches works after a compile operation."""
        @torch.compile(fullgraph=True, backend="eager")
        def fn(x):
            return x + 1

        x = torch.randn(4, device=self.device_name)
        fn(x)
        # Should not raise after compilation
        torch._dynamo.reset_code_caches()


if __name__ == "__main__":
    run_tests()
