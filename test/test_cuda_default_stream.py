# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.default_stream (via torch.npu.default_stream) 接口功能正确性
API 名称：torch.cuda.default_stream / torch.npu.default_stream
API 签名：torch.npu.default_stream(device=None) -> Stream

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 返回 Stream 对象                    | 已覆盖                  |
| device 参数      | 传入设备索引                        | 已覆盖                  |

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


class TestCudaDefaultStream(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_returns_stream(self):
        """Verify default_stream returns a Stream instance."""
        s = torch.npu.default_stream()
        self.assertIsInstance(s, torch.npu.Stream)

    def test_npu_with_device_arg(self):
        """Verify default_stream accepts a device index argument."""
        s = torch.npu.default_stream(0)
        self.assertIsInstance(s, torch.npu.Stream)


if __name__ == "__main__":
    run_tests()
