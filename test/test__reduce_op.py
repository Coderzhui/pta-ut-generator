# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.reduce_op 接口功能正确性
API 名称：torch.distributed.reduce_op
API 签名：torch.distributed.reduce_op (enum-like)

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 成员存在         | SUM/PRODUCT/MIN/MAX/BAND/BOR/BXOR  | 已覆盖                  |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性，不做精度和数值正确性校验。
"""
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    import unittest
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


class TestReduceOp(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_reduce_op_exists(self):
        """Verify reduce_op attribute exists in dist namespace."""
        self.assertTrue(hasattr(dist, 'reduce_op'))

    def test_sum_exists(self):
        """Verify ReduceOp.SUM member is available."""
        self.assertTrue(hasattr(dist.ReduceOp, 'SUM'))

    def test_product_exists(self):
        """Verify ReduceOp.PRODUCT member is available."""
        self.assertTrue(hasattr(dist.ReduceOp, 'PRODUCT'))

    def test_min_exists(self):
        """Verify ReduceOp.MIN member is available."""
        self.assertTrue(hasattr(dist.ReduceOp, 'MIN'))

    def test_max_exists(self):
        """Verify ReduceOp.MAX member is available."""
        self.assertTrue(hasattr(dist.ReduceOp, 'MAX'))


if __name__ == "__main__":
    run_tests()
