# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.use_mem_pool (via torch.npu.use_mem_pool) 接口功能正确性
API 名称：torch.cuda.use_mem_pool / torch.npu.use_mem_pool
API 签名：torch.npu.use_mem_pool(*args, **kwargs)

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 存在于 torch.npu 命名空间           | 已覆盖                  |

未覆盖项及原因：
- 实际使用需特定内存池配置

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


class TestCudaUseMemPool(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_use_mem_pool_callable(self):
        """Verify use_mem_pool is a callable in torch.npu namespace."""
        self.assertTrue(callable(torch.npu.use_mem_pool))


if __name__ == "__main__":
    run_tests()
