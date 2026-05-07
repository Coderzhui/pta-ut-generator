# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.Stream.synchronize (via torch.npu.Stream.synchronize) 接口功能正确性
API 名称：torch.cuda.Stream.synchronize / torch.npu.Stream.synchronize
API 签名：torch.npu.Stream.synchronize() -> None

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 默认流同步返回 None                 | 已覆盖                  |
| 新建流           | 新建流同步不报错                    | 已覆盖                  |
| 多次调用         | 连续调用不报错                      | 已覆盖                  |

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


class TestCudaStreamSynchronize(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_default_stream_synchronize(self):
        """Verify default stream synchronize returns None."""
        s = torch.npu.default_stream()
        result = s.synchronize()
        self.assertIsNone(result)

    def test_npu_new_stream_synchronize(self):
        """Verify synchronize on a new stream completes without error."""
        s = torch.npu.Stream()
        s.synchronize()

    def test_npu_multiple_synchronize(self):
        """Verify consecutive synchronize calls do not raise."""
        s = torch.npu.Stream()
        s.synchronize()
        s.synchronize()


if __name__ == "__main__":
    run_tests()
