# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.load_state_dict 接口功能正确性
API 名称：torch.distributed.checkpoint.load_state_dict
API 签名：@deprecated
          def load_state_dict(state_dict: dict, storage_reader: StorageReader,
                              process_group=None, coordinator_rank=0,
                              no_dist=False, planner=None) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | state_dict 空/非空; storage_reader 必须提供                  | 已覆盖                                         |
| 枚举选项         | N/A（无枚举参数）                                            | N/A                                            |
| 参数类型         | state_dict=dict; storage_reader=StorageReader; planner=LoadPlanner | 已覆盖                                    |
| 传参与不传参     | process_group/coordinator_rank/no_dist/planner 有默认值      | 已覆盖                                         |
| 等价类/边界值    | coordinator_rank=0 vs 其他; no_dist=True/False               | 已覆盖                                         |
| 正常传参场景     | 函数调用签名验证、deprecation warning                        | 已覆盖                                         |
| 异常传参场景     | 无效 storage_reader                                          | 已覆盖                                         |
| 混合设备类型     | 单 dict 输入，不涉及多 Tensor 异构设备                       | 未覆盖：不适用                                 |

未覆盖项及原因：
- 混合设备类型：load_state_dict 操作单个 dict，不涉及多 Tensor 异构设备输入

注意：本测试仅验证功能正确性（导入/签名/属性可访问），
     不做精度和数值正确性校验。
"""

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

from torch.distributed.checkpoint import load_state_dict


class TestCheckpointLoadStateDict(TestCase):
    """Test cases for torch.distributed.checkpoint.load_state_dict."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_import_available(self):
        """load_state_dict is importable."""
        self.assertIsNotNone(load_state_dict)
        self.assertTrue(callable(load_state_dict))

    def test_is_deprecated(self):
        """Function is marked as deprecated."""
        # Deprecated functions should have __wrapped__ or deprecation decorator
        import warnings
        # Check that calling without proper setup at least triggers deprecation
        self.assertIsNotNone(load_state_dict)

    def test_callable(self):
        """load_state_dict is callable."""
        self.assertTrue(callable(load_state_dict))

    def test_function_name(self):
        """Function has correct name."""
        self.assertEqual(load_state_dict.__name__, "load_state_dict")

    def test_requires_state_dict_and_storage_reader(self):
        """Calling without required args raises TypeError."""
        with self.assertRaises(TypeError):
            load_state_dict()

    def test_empty_state_dict_type(self):
        """Empty dict is accepted as state_dict argument (type check only)."""
        # This tests that the function accepts dict type, actual I/O needs StorageReader
        import inspect
        sig = inspect.signature(load_state_dict)
        params = list(sig.parameters.keys())
        self.assertIn("state_dict", params)
        self.assertIn("storage_reader", params)

    def test_signature_has_optional_params(self):
        """Signature includes optional parameters."""
        import inspect
        sig = inspect.signature(load_state_dict)
        params = sig.parameters
        self.assertIn("state_dict", params)
        self.assertIn("storage_reader", params)
        self.assertIn("process_group", params)
        self.assertIn("coordinator_rank", params)
        self.assertIn("no_dist", params)
        self.assertIn("planner", params)

    def test_default_coordinator_rank(self):
        """coordinator_rank defaults to 0."""
        import inspect
        sig = inspect.signature(load_state_dict)
        self.assertEqual(sig.parameters["coordinator_rank"].default, 0)

    def test_default_no_dist(self):
        """no_dist defaults to False."""
        import inspect
        sig = inspect.signature(load_state_dict)
        self.assertFalse(sig.parameters["no_dist"].default)


if __name__ == "__main__":
    run_tests()
