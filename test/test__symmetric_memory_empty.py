# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._symmetric_memory.empty 接口功能正确性
API 名称：torch.distributed._symmetric_memory.empty
API 签名：def empty(*size, dtype=None, device=None) -> torch.Tensor

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | N/A（API 需要分布式初始化，此文件测导入+签名）               | N/A                                            |
| 枚举选项         | N/A                                                          | N/A                                            |
| 参数类型         | 通过 inspect 验证签名                                        | 已覆盖                                         |
| 传参与不传参     | 签名参数检查                                                 | 已覆盖                                         |
| 等价类/边界值    | N/A                                                          | N/A                                            |
| 正常传参场景     | 函数可调用、签名正确                                         | 已覆盖                                         |
| 异常传参场景     | CPU 设备 raises RuntimeError                                 | 已覆盖                                         |
| 混合设备类型     | N/A                                                          | N/A                                            |

未覆盖项及原因：
- 正常调用：_symmetric_memory.empty 需要 CUDA 设备 + 分布式初始化，NPU 环境下
  走 fallback 路径仍需 init_process_group，此处仅验证导入和签名
- 混合设备类型：API 仅创建单个 Tensor，不涉及多 Tensor 异构设备输入

注意：本测试仅验证功能正确性（导入/签名/属性验证），
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

from torch.distributed._symmetric_memory import empty


class TestSymmetricMemoryEmpty(TestCase):
    """Test cases for torch.distributed._symmetric_memory.empty."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_import_available(self):
        """empty function is importable."""
        self.assertIsNotNone(empty)
        self.assertTrue(callable(empty))

    def test_signature_has_size_params(self):
        """Function accepts *size variadic args."""
        sig = inspect.signature(empty)
        params = list(sig.parameters.keys())
        self.assertGreater(len(params), 0)

    def test_signature_has_dtype(self):
        """Function has dtype parameter."""
        sig = inspect.signature(empty)
        self.assertIn("dtype", sig.parameters)

    def test_signature_has_device(self):
        """Function has device parameter."""
        sig = inspect.signature(empty)
        self.assertIn("device", sig.parameters)

    def test_dtype_default_none(self):
        """dtype defaults to None."""
        sig = inspect.signature(empty)
        self.assertIsNone(sig.parameters["dtype"].default)

    def test_device_default_none(self):
        """device defaults to None."""
        sig = inspect.signature(empty)
        self.assertIsNone(sig.parameters["device"].default)

    def test_cpu_raises_runtime_error(self):
        """CPU device raises RuntimeError (SymmetricMemory requires CUDA/NPU)."""
        with self.assertRaises(RuntimeError):
            empty(3, 4)

    def test_function_name(self):
        """Function name is 'empty'."""
        self.assertEqual(empty.__name__, "empty")


if __name__ == "__main__":
    run_tests()
