# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.CUDAGraph.pool (via torch.npu.NPUGraph.pool) 属性功能正确性
API 名称：torch.cuda.CUDAGraph.pool / torch.npu.NPUGraph.pool
API 签名：NPUGraph.pool (property)

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 属性访问         | pool 属性可访问                     | 已覆盖                  |

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


class TestCudaCUDAGraphPool(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_pool_attribute(self):
        """Verify pool property is accessible (may be None initially)."""
        g = torch.npu.NPUGraph()
        _ = g.pool


if __name__ == "__main__":
    run_tests()
