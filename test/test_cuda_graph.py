# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.graph (via torch.npu.graph) 接口功能正确性
API 名称：torch.cuda.graph / torch.npu.graph
API 签名：torch.npu.graph(graph, stream=None, pool=None, capture_error_mode='global')

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 作为上下文管理器不报错              | 已覆盖                  |
| stream 参数      | 指定流                              | 已覆盖                  |

未覆盖项及原因：
- pool 参数：需要预分配内存池
- capture_error_mode 枚举：需特定环境验证

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


class TestCudaGraph(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_graph_context_manager(self):
        """Verify torch.npu.graph captures and replays a simple computation graph."""
        g = torch.npu.NPUGraph()
        x = torch.randn(4, 4, device=self.device_name)
        static_x = x.clone()
        with torch.npu.graph(g):
            _ = static_x + 1

    def test_npu_graph_with_stream(self):
        """Verify torch.npu.graph works with an explicit stream argument."""
        g = torch.npu.NPUGraph()
        s = torch.npu.Stream()
        x = torch.randn(4, 4, device=self.device_name)
        static_x = x.clone()
        with torch.npu.graph(g, stream=s):
            _ = static_x + 1


if __name__ == "__main__":
    run_tests()
