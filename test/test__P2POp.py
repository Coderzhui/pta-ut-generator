# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.P2POp 接口功能正确性
API 名称：torch.distributed.P2POp
API 签名：torch.distributed.P2POp(op, tensor, peer=None, group=None, tag=0)

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 类存在           | P2POp 类可导入                      | 已覆盖                  |

未覆盖项及原因：
- 实例化需 init_process_group 先完成

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


class TestP2POp(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_p2pop_class_exists(self):
        """Verify P2POp class is available in dist namespace."""
        self.assertTrue(hasattr(dist, 'P2POp'))

    def test_p2pop_is_class(self):
        """Verify P2POp is a class (type)."""
        self.assertIsInstance(dist.P2POp, type)


if __name__ == "__main__":
    run_tests()
