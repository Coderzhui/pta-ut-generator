# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.CUDAGraph.capture_end (via torch.npu.NPUGraph.capture_end) 接口功能正确性
API 名称：torch.cuda.CUDAGraph.capture_end / torch.npu.NPUGraph.capture_end
API 签名：NPUGraph.capture_end()

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 在 capture_begin 后调用不报错       | 已覆盖                  |

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


class TestCudaCUDAGraphCaptureEnd(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_capture_end(self):
        """Verify capture_end stops graph capture without error."""
        g = torch.npu.NPUGraph()
        x = torch.randn(4, 4, device=self.device_name)
        static_x = x.clone()
        with torch.npu.stream(torch.npu.Stream()):
            g.capture_begin()
            _ = static_x + 1
            g.capture_end()


if __name__ == "__main__":
    run_tests()
