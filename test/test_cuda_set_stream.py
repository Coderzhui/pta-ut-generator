# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.set_stream (via torch.npu.set_stream) 接口功能正确性
API 名称：torch.cuda.set_stream / torch.npu.set_stream
API 签名：torch.npu.set_stream(stream)

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 设置流不报错                        | 已覆盖                  |
| 恢复默认流       | 切换回默认流不报错                  | 已覆盖                  |

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


class TestCudaSetStream(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")
        self._orig_stream = torch.npu.current_stream()

    def tearDown(self):
        torch.npu.set_stream(self._orig_stream)
        super().tearDown()

    def test_npu_set_stream(self):
        """Verify set_stream switches to a new stream without error."""
        s = torch.npu.Stream()
        torch.npu.set_stream(s)

    def test_npu_set_default_stream(self):
        """Verify set_stream can switch back to the default stream."""
        default = torch.npu.default_stream()
        torch.npu.set_stream(default)


if __name__ == "__main__":
    run_tests()
