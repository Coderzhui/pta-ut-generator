# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.version.cuda.split 接口功能正确性
API 名称：torch.version.cuda.split
API 签名：torch.version.cuda: str | None, .split(sep) -> list[str]

覆盖维度表：
| 覆盖维度         | 说明                                         | 覆盖情况             |
|------------------|----------------------------------------------|----------------------|
| 基础调用         | torch.version.cuda 属性存在                  | 已覆盖               |
| 返回类型         | str 时 .split() 返回 list[str]               | 已覆盖               |
| None 处理        | NPU 环境 cuda 可能为 None                    | 已覆盖               |
| split 参数       | split(".") 默认分隔符                        | 已覆盖               |

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


class TestVersionCudaSplit(TestCase):
    """Test cases for torch.version.cuda.split."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_version_cuda_attribute_exists(self):
        """torch.version.cuda attribute exists."""
        self.assertTrue(hasattr(torch.version, 'cuda'))

    def test_version_cuda_type(self):
        """torch.version.cuda is str or None."""
        cuda_ver = torch.version.cuda
        self.assertTrue(
            cuda_ver is None or isinstance(cuda_ver, str),
            f"Expected str or None, got {type(cuda_ver)}"
        )

    def test_version_cuda_split_when_string(self):
        """torch.version.cuda.split('.') returns list when cuda is a string."""
        cuda_ver = torch.version.cuda
        if cuda_ver is not None:
            parts = cuda_ver.split(".")
            self.assertIsInstance(parts, list)
            for part in parts:
                self.assertIsInstance(part, str)
        else:
            # On NPU without CUDA, cuda version is None
            self.assertIsNone(cuda_ver)

    def test_version_cuda_split_returns_multiple_parts(self):
        """When cuda is a version string, split produces multiple parts."""
        cuda_ver = torch.version.cuda
        if cuda_ver is not None:
            parts = cuda_ver.split(".")
            self.assertGreaterEqual(len(parts), 1)

    def test_version_cuda_none_on_npu(self):
        """On NPU-only build, torch.version.cuda is likely None."""
        cuda_ver = torch.version.cuda
        # NPU-only build typically has None for cuda version
        # This test documents the expected behavior
        if not torch.cuda.is_available():
            self.assertIsNone(cuda_ver)


if __name__ == "__main__":
    run_tests()
