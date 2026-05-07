# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.MemPool (via torch.npu.MemPool) 接口功能正确性
API 名称：torch.cuda.MemPool / torch.npu.MemPool, MemPool.allocator
API 签名：torch.npu.MemPool()

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础创建         | 创建 MemPool 不报错                 | 已覆盖                  |
| allocator 属性   | allocator 属性可访问                | 已覆盖                  |

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


class TestCudaMemPoolAllocator(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_create_mempool(self):
        """Verify MemPool instance can be created without error."""
        p = torch.npu.MemPool()
        self.assertIsInstance(p, torch.npu.MemPool)

    def test_npu_allocator_attribute(self):
        """Verify allocator property is accessible on MemPool."""
        p = torch.npu.MemPool()
        _ = p.allocator


if __name__ == "__main__":
    run_tests()
