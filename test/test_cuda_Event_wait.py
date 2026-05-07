# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.Event.wait (via torch.npu.Event.wait) 接口功能正确性
API 名称：torch.cuda.Event.wait / torch.npu.Event.wait
API 签名：Event.wait(stream=None)

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | wait 不报错                         | 已覆盖                  |

未覆盖项及原因：
- stream 参数：需构造特定流场景

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


class TestCudaEventWait(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_event_wait(self):
        """Verify wait() completes without error."""
        e = torch.npu.Event()
        e.wait()


if __name__ == "__main__":
    run_tests()
