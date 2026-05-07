# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.Event (via torch.npu.Event) 接口功能正确性
API 名称：torch.cuda.Event / torch.npu.Event
API 签名：torch.npu.Event(*, enable_timing=False, blocking=False, interprocess=False)

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础创建         | 无参数创建 Event 不报错             | 已覆盖                  |
| 参数枚举         | enable_timing/blocking 参数组合     | 已覆盖                  |

未覆盖项及原因：
- interprocess=True：需要多进程环境
- 方法级测试（query/wait）已拆分为独立文件

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


class TestCudaEvent(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_create_event(self):
        """Verify Event instance can be created with default args."""
        e = torch.npu.Event()
        self.assertIsInstance(e, torch.npu.Event)

    def test_npu_event_enable_timing(self):
        """Verify Event with enable_timing=True can be created."""
        e = torch.npu.Event(enable_timing=True)
        self.assertIsInstance(e, torch.npu.Event)

    def test_npu_event_blocking(self):
        """Verify Event with blocking=True can be created."""
        e = torch.npu.Event(blocking=True)
        self.assertIsInstance(e, torch.npu.Event)

    def test_npu_event_record(self):
        """Verify record() on a stream completes without error."""
        e = torch.npu.Event()
        s = torch.npu.default_stream()
        e.record(s)


if __name__ == "__main__":
    run_tests()
