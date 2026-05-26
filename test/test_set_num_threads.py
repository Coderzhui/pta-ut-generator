# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.set_num_threads 接口功能正确性
API 名称：torch.set_num_threads
API 签名：def set_num_threads(nthreads: int) -> None

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况             |
|------------------|--------------------------------------------|----------------------|
| 基础调用         | 正常 int 参数调用不报错                    | 已覆盖               |
| 返回类型         | 返回 None                                  | 已覆盖               |
| 参数边界值       | nthreads=1 / 典型值                        | 已覆盖               |
| 状态变更         | 设置后 get_num_threads 返回正确值          | 已覆盖               |
| 幂等性           | 多次设置不报错                             | 已覆盖               |
| 异常路径         | 无稳定异常路径                             | 未覆盖：内部无校验   |

未覆盖项及原因：
- 异常路径：set_num_threads 内部无参数校验

注意：本测试仅验证功能正确性（调用不报错、线程数状态符合预期），
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


class TestSetNumThreads(TestCase):
    """Test cases for torch.set_num_threads."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )
        self._original_threads = torch.get_num_threads()

    def tearDown(self):
        torch.set_num_threads(self._original_threads)
        super().tearDown()

    def test_set_num_threads_returns_none(self):
        """set_num_threads returns None."""
        result = torch.set_num_threads(4)
        self.assertIsNone(result)

    def test_set_num_threads_1(self):
        """set_num_threads(1) sets thread count to 1."""
        torch.set_num_threads(1)
        self.assertEqual(torch.get_num_threads(), 1)

    def test_set_num_threads_4(self):
        """set_num_threads(4) sets thread count to 4."""
        torch.set_num_threads(4)
        self.assertEqual(torch.get_num_threads(), 4)

    def test_set_num_threads_idempotent(self):
        """Calling set_num_threads multiple times works."""
        torch.set_num_threads(2)
        torch.set_num_threads(2)
        self.assertEqual(torch.get_num_threads(), 2)

    def test_set_num_threads_callable(self):
        """set_num_threads is callable."""
        self.assertTrue(callable(torch.set_num_threads))


if __name__ == "__main__":
    run_tests()
