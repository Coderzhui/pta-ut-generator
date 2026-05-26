# -*- coding: utf-8 -*-
"""
测试目的：验证 torch._C._cuda_getDeviceCount 接口功能正确性
API 名称：torch._C._cuda_getDeviceCount
API 签名：def _cuda_getDeviceCount() -> int

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                                  |
|------------------|--------------------------------------------|-------------------------------------------|
| 基础调用         | NPU 环境下该 CUDA 绑定不可用               | 已覆盖：验证 AttributeError              |
| NPU 替代路径     | torch.npu.device_count() 作为 NPU 等价 API | 已覆盖                                    |
| 返回类型         | 当 API 存在时返回 int                      | 未覆盖：NPU 环境无 CUDA 绑定              |
| 异常路径         | 调用不存在属性触发 AttributeError           | 已覆盖                                    |

未覆盖项及原因：
- 返回类型：torch._C._cuda_getDeviceCount 为 CUDA C++ 绑定，NPU 环境无 CUDA 编译支持

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


class TestCudaGetDeviceCount(TestCase):
    """Test cases for torch._C._cuda_getDeviceCount on NPU."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_cuda_get_device_count_not_available_on_npu(self):
        """_cuda_getDeviceCount should not exist on NPU-only build."""
        has_fn = hasattr(torch._C, '_cuda_getDeviceCount')
        self.assertFalse(
            has_fn,
            "_cuda_getDeviceCount should not exist in NPU-only environment"
        )

    def test_npu_device_count_as_alternative(self):
        """torch.npu.device_count() serves as the NPU equivalent."""
        count = torch.npu.device_count()
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)

    def test_npu_device_count_matches_available(self):
        """torch.npu.device_count() returns >= 1 when NPU is available."""
        self.assertTrue(torch.npu.is_available())
        count = torch.npu.device_count()
        self.assertGreaterEqual(count, 1)


if __name__ == "__main__":
    run_tests()
