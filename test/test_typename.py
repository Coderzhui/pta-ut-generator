# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.typename 接口功能正确性
API 名称：torch.typename
API 签名：def typename(obj: Any, /) -> str

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况             |
|------------------|--------------------------------------------|----------------------|
| 基础调用         | 对各种类型对象调用不报错                   | 已覆盖               |
| 返回类型         | 始终返回 str                               | 已覆盖               |
| 参数类型         | Tensor / int / float / str / list / module | 已覆盖               |
| NPU Tensor       | NPU 设备上的 Tensor                        | 已覆盖               |
| 空/非空          | None 作为输入                              | 已覆盖               |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、返回类型符合预期），
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


class TestTypename(TestCase):
    """Test cases for torch.typename."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_typename_tensor(self):
        """typename of a Tensor returns str."""
        x = torch.randn(3, 4)
        result = torch.typename(x)
        self.assertIsInstance(result, str)
        self.assertIn("Tensor", result)

    def test_typename_npu_tensor(self):
        """typename of an NPU Tensor returns str."""
        x = torch.randn(3, 4, device=self.device_name)
        result = torch.typename(x)
        self.assertIsInstance(result, str)

    def test_typename_int(self):
        """typename of int returns str."""
        result = torch.typename(42)
        self.assertIsInstance(result, str)
        self.assertIn("int", result)

    def test_typename_float(self):
        """typename of float returns str."""
        result = torch.typename(3.14)
        self.assertIsInstance(result, str)
        self.assertIn("float", result)

    def test_typename_str(self):
        """typename of str returns str."""
        result = torch.typename("hello")
        self.assertIsInstance(result, str)
        self.assertIn("str", result)

    def test_typename_list(self):
        """typename of list returns str."""
        result = torch.typename([1, 2, 3])
        self.assertIsInstance(result, str)
        self.assertIn("list", result)

    def test_typename_none(self):
        """typename of None returns str."""
        result = torch.typename(None)
        self.assertIsInstance(result, str)
        self.assertIn("NoneType", result)

    def test_typename_module(self):
        """typename of a module returns str."""
        result = torch.typename(torch)
        self.assertIsInstance(result, str)

    def test_typename_callable(self):
        """torch.typename is callable."""
        self.assertTrue(callable(torch.typename))


if __name__ == "__main__":
    run_tests()
