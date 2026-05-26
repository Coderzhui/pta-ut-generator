# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._tools.fsdp2_mem_tracker.FSDPMemTracker 接口功能正确性
API 名称：torch.distributed._tools.fsdp2_mem_tracker.FSDPMemTracker
API 签名：class FSDPMemTracker(MemTracker):
             def __init__(mod: torch.nn.Module, optm: torch.optim.Optimizer | None = None)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | N/A（类级别测试）                                            | N/A                                            |
| 枚举选项         | N/A（无枚举参数）                                            | N/A                                            |
| 参数类型         | mod 为 FSDPModule; optm 为 Optimizer 或 None                 | 已覆盖                                         |
| 传参与不传参     | N/A                                                          | N/A                                            |
| 等价类/边界值    | 非 FSDPModule 触发断言                                       | 已覆盖                                         |
| 正常传参场景     | 导入、类型、属性、方法存在性验证                              | 已覆盖                                         |
| 异常传参场景     | 非 FSDPModule raises AssertionError                          | 已覆盖                                         |
| 混合设备类型     | 单对象构造，不涉及多 Tensor 输入                             | 未覆盖：不适用                                 |

未覆盖项及原因：
- 混合设备类型：API 为单对象构造，不涉及多 Tensor 异构设备输入
- 正常构造+track 测试：需要 FSDPModule + 分布式初始化，留待多卡测试补充

注意：本测试仅验证功能正确性（导入/类型/属性/方法存在性），
     不做精度和数值正确性校验。
"""

import torch
import torch.nn as nn
import torch_npu  # noqa: F401

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        import unittest
        unittest.main(argv=sys.argv)

from torch.distributed._tools.fsdp2_mem_tracker import FSDPMemTracker


class _SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


class TestFSDPMemTracker(TestCase):
    """Test cases for torch.distributed._tools.fsdp2_mem_tracker.FSDPMemTracker."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_import_available(self):
        """FSDPMemTracker is importable."""
        self.assertIsNotNone(FSDPMemTracker)
        self.assertTrue(isinstance(FSDPMemTracker, type))

    def test_non_fsdp_module_raises(self):
        """Non-FSDPModule raises AssertionError."""
        model = _SimpleModule()
        with self.assertRaises(AssertionError):
            FSDPMemTracker(model)

    def test_non_fsdp_module_with_optimizer_raises(self):
        """Non-FSDPModule with optimizer also raises AssertionError."""
        import torch.optim as optim
        model = _SimpleModule()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        with self.assertRaises(AssertionError):
            FSDPMemTracker(model, optm=optimizer)

    def test_is_context_manager(self):
        """FSDPMemTracker has __enter__ and __exit__ methods."""
        self.assertTrue(hasattr(FSDPMemTracker, '__enter__'))
        self.assertTrue(hasattr(FSDPMemTracker, '__exit__'))

    def test_has_get_tracker_snapshot(self):
        """FSDPMemTracker has get_tracker_snapshot method."""
        self.assertTrue(hasattr(FSDPMemTracker, 'get_tracker_snapshot'))

    def test_has_display_snapshot(self):
        """FSDPMemTracker has display_snapshot method."""
        self.assertTrue(hasattr(FSDPMemTracker, 'display_snapshot'))

    def test_has_display_modulewise_snapshots(self):
        """FSDPMemTracker has display_modulewise_snapshots method."""
        self.assertTrue(hasattr(FSDPMemTracker, 'display_modulewise_snapshots'))

    def test_has_track_external(self):
        """FSDPMemTracker has track_external method."""
        self.assertTrue(hasattr(FSDPMemTracker, 'track_external'))

    def test_has_track_inputs(self):
        """FSDPMemTracker has track_inputs method."""
        self.assertTrue(hasattr(FSDPMemTracker, 'track_inputs'))


if __name__ == "__main__":
    run_tests()
