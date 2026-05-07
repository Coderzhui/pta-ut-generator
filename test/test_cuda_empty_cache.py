# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.empty_cache 接口功能正确性
API 名称：torch.cuda.empty_cache
API 签名：torch.cuda.empty_cache() -> None

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 无参数调用返回 None                 | 已覆盖                  |
| 多次调用         | 连续调用不报错                      | 已覆盖                  |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性，不做精度和数值正确性校验。
"""
import torch
import torch_npu  # noqa: F401

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    import unittest
    from unittest import TestCase
    def run_tests():
        unittest.main(argv=sys.argv)


class TestCudaEmptyCache(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_returns_none(self):
        """Verify empty_cache returns None."""
        result = torch.cuda.empty_cache()
        self.assertIsNone(result)

    def test_npu_multiple_calls(self):
        """Verify consecutive empty_cache calls do not raise."""
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_tests()
