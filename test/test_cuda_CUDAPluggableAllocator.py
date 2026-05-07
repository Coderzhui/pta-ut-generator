# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.CUDAPluggableAllocator 接口在 NPU 环境的可用性
API 名称：torch.cuda.CUDAPluggableAllocator
API 签名：torch.cuda.CUDAPluggableAllocator(alloc_fn, free_fn)

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | API 是否存在                        | 未覆盖：torch_npu 无此 API |

未覆盖项及原因：
- torch.npu 命名空间下不存在 CUDAPluggableAllocator，该 API 在 NPU 环境不可用

注意：本测试仅验证 API 存在性。
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


class TestCudaCUDAPluggableAllocator(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_no_pluggable_allocator(self):
        """CUDAPluggableAllocator is not available in torch_npu."""
        self.assertFalse(hasattr(torch.npu, 'CUDAPluggableAllocator'))


if __name__ == "__main__":
    run_tests()
