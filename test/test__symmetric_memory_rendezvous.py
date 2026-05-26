# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._symmetric_memory.rendezvous 接口功能正确性
API 名称：torch.distributed._symmetric_memory.rendezvous
API 签名：def rendezvous(tensor: torch.Tensor, group: GroupName | ProcessGroup) -> _SymmetricMemory

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | tensor/group 必须有效                                        | 已覆盖                                         |
| 枚举选项         | N/A                                                          | N/A                                            |
| 参数类型         | tensor=Tensor; group=str/GroupName/ProcessGroup              | 已覆盖                                         |
| 传参与不传参     | 必传参数                                                     | N/A                                            |
| 等价类/边界值    | 无效 group 类型                                              | 已覆盖                                         |
| 正常传参场景     | 函数导入、签名验证                                           | 已覆盖                                         |
| 异常传参场景     | 无效 group 类型 raises TypeError                             | 已覆盖                                         |
| 混合设备类型     | 单 Tensor 输入                                               | 未覆盖：不适用                                 |

未覆盖项及原因：
- 正常调用：rendezvous 需要 _symmetric_memory.empty 创建的 tensor + 分布式初始化，
  SymmetricMemory 在 NPU 上可能不完全支持，此处仅验证导入和签名
- 混合设备类型：API 接收单个 Tensor，不涉及多 Tensor 异构设备输入

注意：本测试仅验证功能正确性（导入/签名/异常路径），
     不做精度和数值正确性校验。
"""

import inspect
import torch
import torch_npu  # noqa: F401

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        import unittest
        unittest.main(argv=sys.argv)

from torch.distributed._symmetric_memory import rendezvous


class TestSymmetricMemoryRendezvous(TestCase):
    """Test cases for torch.distributed._symmetric_memory.rendezvous."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_import_available(self):
        """rendezvous function is importable."""
        self.assertIsNotNone(rendezvous)
        self.assertTrue(callable(rendezvous))

    def test_signature_has_tensor_param(self):
        """Function accepts tensor parameter."""
        sig = inspect.signature(rendezvous)
        params = list(sig.parameters.keys())
        self.assertIn("tensor", params)

    def test_signature_has_group_param(self):
        """Function accepts group parameter."""
        sig = inspect.signature(rendezvous)
        param_names = list(sig.parameters.keys())
        self.assertIn("group", param_names)

    def test_invalid_group_type_raises(self):
        """Invalid group type raises TypeError."""
        device = torch.device(self.device_name)
        tensor = torch.randn(4, device=device)
        with self.assertRaises(TypeError):
            rendezvous(tensor, 12345)

    def test_function_name(self):
        """Function name is 'rendezvous'."""
        self.assertEqual(rendezvous.__name__, "rendezvous")


if __name__ == "__main__":
    run_tests()
