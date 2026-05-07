# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.Stream.query (via torch.npu.Stream.query) 接口功能正确性
API 名称：torch.cuda.Stream.query / torch.npu.Stream.query
API 签名：torch.npu.Stream.query() -> bool

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 返回 bool 类型                      | 已覆盖                  |
| 默认流           | 默认流上 query 返回 True            | 已覆盖                  |
| 新建流           | 新建流上 query 不报错               | 已覆盖                  |

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


class TestCudaStreamQuery(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_returns_bool(self):
        """Verify Stream.query returns a boolean."""
        s = torch.npu.Stream()
        result = s.query()
        self.assertIsInstance(result, bool)

    def test_npu_default_stream_query(self):
        """Verify default stream query returns True (idle)."""
        s = torch.npu.default_stream()
        result = s.query()
        self.assertTrue(result)

    def test_npu_new_stream_query(self):
        """Verify query on a newly created stream returns bool."""
        s = torch.npu.Stream()
        result = s.query()
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    run_tests()
