# -*- coding: utf-8 -*-
"""
测试目的：验证 torch._subclasses.FakeTensorMode 导入路径正确性
API 名称：torch._subclasses.FakeTensorMode
API 签名：同 torch._subclasses.fake_tensor.FakeTensorMode（同一类的 re-export）

覆盖维度表：
| 覆盖维度         | 说明                                           | 覆盖情况             |
|------------------|------------------------------------------------|----------------------|
| 导入路径         | torch._subclasses.FakeTensorMode 可导入        | 已覆盖               |
| 等价性           | 与 torch._subclasses.fake_tensor.FakeTensorMode 为同一对象 | 已覆盖    |
| 基础实例化       | 通过此路径实例化不报错                          | 已覆盖               |
| 上下文管理器     | with 语句正常工作                              | 已覆盖               |

未覆盖项及原因：
- 无（完整功能测试已在 _subclasses_fake_tensor_FakeTensorMode 中覆盖）

注意：本测试仅验证导入路径等价性及基本实例化，
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


class TestSubclassesFakeTensorMode(TestCase):
    """Test cases for torch._subclasses.FakeTensorMode import path."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_import_from_subclasses(self):
        """FakeTensorMode is accessible via torch._subclasses."""
        from torch._subclasses import FakeTensorMode
        self.assertIsNotNone(FakeTensorMode)

    def test_same_as_fake_tensor_module(self):
        """torch._subclasses.FakeTensorMode is the same class as the canonical path."""
        from torch._subclasses import FakeTensorMode as FT_short
        from torch._subclasses.fake_tensor import FakeTensorMode as FT_long
        self.assertIs(FT_short, FT_long)

    def test_instantiate_via_short_path(self):
        """Can instantiate FakeTensorMode via the short import path."""
        from torch._subclasses import FakeTensorMode
        mode = FakeTensorMode()
        self.assertIsInstance(mode, FakeTensorMode)

    def test_context_manager_via_short_path(self):
        """FakeTensorMode from short path works as context manager."""
        from torch._subclasses import FakeTensorMode
        with FakeTensorMode() as mode:
            self.assertIsInstance(mode, FakeTensorMode)


if __name__ == "__main__":
    run_tests()
