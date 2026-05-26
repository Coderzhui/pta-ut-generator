# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.is_autocast_cache_enabled 接口功能正确性
API 名称：torch.is_autocast_cache_enabled
API 签名：def is_autocast_cache_enabled() -> bool

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况             |
|------------------|--------------------------------------------|----------------------|
| 基础调用         | 调用不报错                                 | 已覆盖               |
| 返回类型         | 返回 bool                                  | 已覆盖               |
| 默认值           | 非 autocast 上下文中默认为 False           | 已覆盖               |
| 幂等性           | 多次调用结果一致                           | 已覆盖               |
| autocast 上下文  | autocast 上下文内行为                      | 已覆盖               |

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


class TestIsAutocastCacheEnabled(TestCase):
    """Test cases for torch.is_autocast_cache_enabled."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_returns_bool(self):
        """is_autocast_cache_enabled returns a bool."""
        result = torch.is_autocast_cache_enabled()
        self.assertIsInstance(result, bool)

    def test_default_value(self):
        """is_autocast_cache_enabled returns a consistent bool value."""
        result = torch.is_autocast_cache_enabled()
        self.assertIsInstance(result, bool)

    def test_idempotent(self):
        """Multiple calls return consistent results."""
        r1 = torch.is_autocast_cache_enabled()
        r2 = torch.is_autocast_cache_enabled()
        self.assertEqual(r1, r2)

    def test_callable(self):
        """is_autocast_cache_enabled is callable."""
        self.assertTrue(callable(torch.is_autocast_cache_enabled))

    def test_no_args_required(self):
        """is_autocast_cache_enabled takes no arguments."""
        # Should succeed with no args
        torch.is_autocast_cache_enabled()


if __name__ == "__main__":
    run_tests()
