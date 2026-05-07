# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.recv_object_list 接口存在性
API 名称：torch.distributed.recv_object_list
API 签名：torch.distributed.recv_object_list(num_objects, src=None, group=None, tag=0)

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 函数存在         | recv_object_list 可导入             | 已覆盖                  |

未覆盖项及原因：
- 实际 send/recv 需 init_process_group 且 P2P 支持

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


class TestRecvObjectList(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_function_exists(self):
        """Verify recv_object_list exists and is callable in dist namespace."""
        self.assertTrue(hasattr(dist, 'recv_object_list'))
        self.assertTrue(callable(dist.recv_object_list))


if __name__ == "__main__":
    run_tests()
