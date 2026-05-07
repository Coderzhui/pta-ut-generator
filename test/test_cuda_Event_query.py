# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.Event.query (via torch.npu.Event.query) 接口功能正确性
API 名称：torch.cuda.Event.query / torch.npu.Event.query
API 签名：Event.query() -> bool

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 返回类型         | query 返回 bool                     | 已覆盖                  |
| record 后       | record + synchronize 后返回 True    | 已覆盖                  |

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


class TestCudaEventQuery(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_event_query_returns_bool(self):
        """Verify query() returns a boolean value."""
        e = torch.npu.Event()
        result = e.query()
        self.assertIsInstance(result, bool)

    def test_npu_event_query_after_record(self):
        """Verify query() returns True after record + synchronize."""
        e = torch.npu.Event()
        s = torch.npu.default_stream()
        e.record(s)
        s.synchronize()
        result = e.query()
        # Event should be marked as completed after stream synchronization
        self.assertTrue(result)


if __name__ == "__main__":
    run_tests()
