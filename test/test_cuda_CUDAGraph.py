# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.CUDAGraph (via torch.npu.NPUGraph) 接口功能正确性
API 名称：torch.cuda.CUDAGraph / torch.npu.NPUGraph
API 签名：torch.npu.NPUGraph()

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础创建         | 创建 NPUGraph 不报错                | 已覆盖                  |

未覆盖项及原因：
- 方法级测试（capture_begin/end/replay/reset/enable_debug_mode/debug_dump/pool）已拆分为独立文件

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


class TestCudaCUDAGraph(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_create_npugraph(self):
        """Verify NPUGraph instance can be created without error."""
        g = torch.npu.NPUGraph()
        self.assertIsInstance(g, torch.npu.NPUGraph)


if __name__ == "__main__":
    run_tests()
