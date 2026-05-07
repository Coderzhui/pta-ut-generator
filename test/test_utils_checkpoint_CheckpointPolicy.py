# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.checkpoint.CheckpointPolicy 接口功能正确性
API 名称：torch.utils.checkpoint.CheckpointPolicy
API 签名：torch.utils.checkpoint.CheckpointPolicy (enum)

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 枚举值存在       | MUST_SAVE, PREFER_SAVE, MUST_RECOMPUTE, PREFER_RECOMPUTE | 已覆盖 |
| 类型             | 枚举成员                            | 已覆盖                  |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性，不做精度和数值正确性校验。
"""
import torch
import torch_npu  # noqa: F401
from torch.utils.checkpoint import CheckpointPolicy

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    import unittest
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


class TestUtilsCheckpointCheckpointPolicy(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_must_save_exists(self):
        """Verify CheckpointPolicy.MUST_SAVE enum member exists."""
        self.assertTrue(hasattr(CheckpointPolicy, 'MUST_SAVE'))

    def test_prefer_save_exists(self):
        """Verify CheckpointPolicy.PREFER_SAVE enum member exists."""
        self.assertTrue(hasattr(CheckpointPolicy, 'PREFER_SAVE'))

    def test_must_recompute_exists(self):
        """Verify CheckpointPolicy.MUST_RECOMPUTE enum member exists."""
        self.assertTrue(hasattr(CheckpointPolicy, 'MUST_RECOMPUTE'))

    def test_prefer_recompute_exists(self):
        """Verify CheckpointPolicy.PREFER_RECOMPUTE enum member exists."""
        self.assertTrue(hasattr(CheckpointPolicy, 'PREFER_RECOMPUTE'))

    def test_policy_values(self):
        """Verify all four policy enum members are distinct."""
        policies = [CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE,
                     CheckpointPolicy.MUST_RECOMPUTE, CheckpointPolicy.PREFER_RECOMPUTE]
        self.assertEqual(len(set(policies)), 4)


if __name__ == "__main__":
    run_tests()
