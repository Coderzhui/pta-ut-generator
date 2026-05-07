# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.Tensor.copy_ 在 NPU 等设备上的原地拷贝功能正确性
API 名称：torch.Tensor.copy_
API 签名：copy_(Tensor self, Tensor src, bool non_blocking=False) -> Tensor

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | N/A                                                          | N/A                                            |
| 枚举选项         | N/A                                                          | N/A                                            |
| 参数类型         | self/src 为 Tensor；non_blocking bool                      | 已覆盖                                         |
| 传参与不传参     | non_blocking 默认 False 与显式 True                        | 已覆盖                                         |
| 等价类/边界值    | 不同 shape（1D/2D）、dtype、同 device                        | 已覆盖                                         |
| 正常传参场景     | 拷贝后 dst shape/dtype/device 与 src 一致                   | 已覆盖                                         |
| 异常传参场景     | device 不一致等（依赖运行时）                               | 未覆盖：错误信息不稳定                         |

未覆盖项及原因：
- 跨 device 非法拷贝：异常类型与文案随版本变化，未做稳定负例

注意：本测试仅验证功能正确性（shape/dtype/device），不做数值精度校验。
"""

import unittest

import torch

try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    pass

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


def _dev_type():
    return torch._C._get_privateuse1_backend_name()


def _npu_available():
    m = getattr(torch, _dev_type(), None)
    return m is not None and getattr(m, "is_available", lambda: False)()


class TestTensorCopy(TestCase):
    def _skip_no_npu(self):
        if not _npu_available():
            self.skipTest(f"{_dev_type()} not available")

    def test_copy_basic_non_blocking_false(self):
        self._skip_no_npu()
        d = torch.device(_dev_type(), 0)
        dst = torch.zeros(4, dtype=torch.float32, device=d)
        src = torch.ones(4, dtype=torch.float32, device=d)
        dst.copy_(src, non_blocking=False)
        self.assertEqual(dst.shape, src.shape)
        self.assertEqual(dst.dtype, src.dtype)
        self.assertEqual(dst.device.type, _dev_type())

    def test_copy_non_blocking_true(self):
        self._skip_no_npu()
        d = torch.device(_dev_type(), 0)
        dst = torch.zeros(8, dtype=torch.float32, device=d)
        src = torch.full((8,), 2.0, dtype=torch.float32, device=d)
        dst.copy_(src, non_blocking=True)
        self.assertEqual(dst.shape, (8,))
        self.assertEqual(dst.dtype, torch.float32)

    def test_copy_bfloat16(self):
        self._skip_no_npu()
        d = torch.device(_dev_type(), 0)
        dst = torch.zeros(3, 3, dtype=torch.bfloat16, device=d)
        src = torch.ones(3, 3, dtype=torch.bfloat16, device=d)
        dst.copy_(src)
        self.assertEqual(dst.shape, (3, 3))
        self.assertEqual(dst.dtype, torch.bfloat16)

    def test_copy_2d(self):
        self._skip_no_npu()
        d = torch.device(_dev_type(), 0)
        dst = torch.zeros(2, 5, dtype=torch.float32, device=d)
        src = torch.randn(2, 5, dtype=torch.float32, device=d)
        dst.copy_(src)
        self.assertEqual(dst.shape, (2, 5))

    def test_copy_cpu_baseline(self):
        dst = torch.zeros(3, dtype=torch.float32)
        src = torch.ones(3, dtype=torch.float32)
        dst.copy_(src)
        self.assertEqual(dst.shape, (3,))
        self.assertEqual(dst.device.type, "cpu")


if __name__ == "__main__":
    run_tests()
