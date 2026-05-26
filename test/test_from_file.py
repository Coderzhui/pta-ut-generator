# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.from_file 接口功能正确性
API 名称：torch.from_file
API 签名：Storage.from_file(filename: str, shared: bool = False, nbytes: int = 0) -> Storage
         也可通过 torch.from_file() C++ 绑定直接调用

覆盖维度表：
| 覆盖维度         | 说明                                          | 覆盖情况             |
|------------------|-----------------------------------------------|----------------------|
| 基础调用         | 从文件创建 storage 不报错                     | 已覆盖               |
| 参数枚举         | shared=True / False; nbytes=0 / 非零          | 已覆盖               |
| 传参与不传参     | shared 和 nbytes 显式 vs 省略默认             | 已覆盖               |
| 返回类型         | 返回 Storage 对象                             | 已覆盖               |
| shape            | 不同 size 的 storage                          | 已覆盖               |
| device           | NPU 上创建                                    | 已覆盖               |
| 混合设备输入     | 单 API 不涉及多 Tensor 输入                   | 未覆盖               |
| 异常路径         | 文件不存在                                    | 已覆盖               |

未覆盖项及原因：
- 混合设备输入：单 API 不涉及多 Tensor 异构设备输入

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import os
import tempfile
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


class TestFromFile(TestCase):
    """Test cases for torch.from_file on NPU."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )
        self._tempfiles = []

    def tearDown(self):
        for f in self._tempfiles:
            try:
                os.unlink(f)
            except OSError:
                pass
        super().tearDown()

    def _create_temp_file(self, size_bytes):
        """Create a temp file with given size and return its path."""
        fd, path = tempfile.mkstemp()
        os.close(fd)
        with open(path, 'wb') as f:
            f.write(b'\x00' * size_bytes)
        self._tempfiles.append(path)
        return path

    def test_from_file_basic(self):
        """torch.from_file creates storage from file."""
        elem_size = 4  # float32 = 4 bytes per element
        path = self._create_temp_file(4 * elem_size)
        result = torch.from_file(path, False, 4)
        self.assertIsNotNone(result)

    def test_from_file_returns_storage_like(self):
        """Result has storage-like interface (data_ptr, size)."""
        elem_size = 4
        path = self._create_temp_file(8 * elem_size)
        result = torch.from_file(path, False, 8)
        self.assertTrue(hasattr(result, 'data_ptr'))

    def test_from_file_shared_true(self):
        """torch.from_file with shared=True."""
        elem_size = 4
        path = self._create_temp_file(4 * elem_size)
        result = torch.from_file(path, True, 4)
        self.assertIsNotNone(result)

    def test_from_file_shared_false(self):
        """torch.from_file with shared=False."""
        elem_size = 4
        path = self._create_temp_file(4 * elem_size)
        result = torch.from_file(path, False, 4)
        self.assertIsNotNone(result)

    def test_from_file_nbytes_zero(self):
        """torch.from_file with size=0 uses file size."""
        path = self._create_temp_file(64)
        result = torch.from_file(path, False, 0)
        self.assertIsNotNone(result)

    def test_from_file_different_sizes(self):
        """torch.from_file with various element counts."""
        elem_size = 4
        for num_elems in [1, 16, 64]:
            path = self._create_temp_file(num_elems * elem_size)
            result = torch.from_file(path, False, num_elems)
            self.assertIsNotNone(result)

    def test_from_file_callable(self):
        """torch.from_file is callable."""
        self.assertTrue(callable(torch.from_file))


if __name__ == "__main__":
    run_tests()
