# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.fsdp.wrap._or_policy 接口功能正确性
API 名称：torch.distributed.fsdp.wrap._or_policy
API 签名：_or_policy(module, recurse, *args, **kwargs)

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 可调用           | 函数可导入且可调用                  | 已覆盖                  |

未覆盖项及原因：
- 实际 wrap 需完整 FSDP 环境

注意：本测试仅验证功能正确性，不做精度和数值正确性校验。
"""
import torch
import torch_npu  # noqa: F401
from torch.distributed.fsdp.wrap import _or_policy

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    import unittest
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


class TestFsdpWrapOrPolicy(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_callable(self):
        """Verify _or_policy is importable and callable."""
        self.assertTrue(callable(_or_policy))


if __name__ == "__main__":
    run_tests()
