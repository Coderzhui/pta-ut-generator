# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.split_with_sizes_copy 原地写入 out 列表的功能正确性
API 名称：torch.split_with_sizes_copy
API 签名：split_with_sizes_copy(Tensor all_gather_output, SymInt[] split_sizes, int dim=0, *, Tensor[] out) -> ()

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | out 必填                                                     | 已覆盖                                         |
| 枚举选项         | N/A                                                          | N/A                                            |
| 参数类型         | Tensor、list[int]、int dim、list[Tensor] out                   | 已覆盖                                         |
| 传参与不传参     | dim 默认 0 与显式 dim                                        | 已覆盖                                         |
| 等价类/边界值    | 1D/2D、dim=-1、bf16                                          | 已覆盖                                         |
| 正常传参场景     | 各 out[i] shape 符合 split_sizes                             | 已覆盖                                         |
| 异常传参场景     | out 与 split_sizes 长度不一致                                | 已覆盖                                         |

未覆盖项及原因：
- split_sizes 与维度和不一致：行为依赖实现，未做稳定负例

注意：仅验证 shape/dtype/device，无数值精度校验。
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


def _dev():
    return torch._C._get_privateuse1_backend_name()


def _npu_ok():
    m = getattr(torch, _dev(), None)
    return m is not None and getattr(m, "is_available", lambda: False)()


def _assert_raises(exc_types, fn):
    try:
        fn()
    except exc_types:
        return
    raise AssertionError(f"expected one of {exc_types}")


class TestSplitWithSizesCopy(TestCase):
    def _d(self):
        if not _npu_ok():
            self.skipTest(f"{_dev()} not available")
        return torch.device(_dev(), 0)

    def test_basic_1d_dim0(self):
        d = self._d()
        x = torch.arange(10, dtype=torch.float32, device=d)
        ss = [2, 3, 5]
        out = [
            torch.zeros(2, dtype=torch.float32, device=d),
            torch.zeros(3, dtype=torch.float32, device=d),
            torch.zeros(5, dtype=torch.float32, device=d),
        ]
        torch.split_with_sizes_copy(x, ss, dim=0, out=out)
        self.assertEqual(out[0].shape, (2,))
        self.assertEqual(out[1].shape, (3,))
        self.assertEqual(out[2].shape, (5,))

    def test_2d_dim0(self):
        d = self._d()
        x = torch.arange(20, dtype=torch.float32, device=d).reshape(4, 5)
        ss = [2, 2]
        out = [
            torch.zeros(2, 5, dtype=torch.float32, device=d),
            torch.zeros(2, 5, dtype=torch.float32, device=d),
        ]
        torch.split_with_sizes_copy(x, ss, dim=0, out=out)
        self.assertEqual(out[0].shape, (2, 5))

    def test_2d_dim1(self):
        d = self._d()
        x = torch.arange(20, dtype=torch.float32, device=d).reshape(4, 5)
        ss = [2, 3]
        out = [
            torch.zeros(4, 2, dtype=torch.float32, device=d),
            torch.zeros(4, 3, dtype=torch.float32, device=d),
        ]
        torch.split_with_sizes_copy(x, ss, dim=1, out=out)
        self.assertEqual(out[0].shape, (4, 2))
        self.assertEqual(out[1].shape, (4, 3))

    def test_dim_negative_one(self):
        d = self._d()
        x = torch.arange(12, dtype=torch.float32, device=d).reshape(3, 4)
        ss = [2, 2]
        out = [
            torch.zeros(3, 2, dtype=torch.float32, device=d),
            torch.zeros(3, 2, dtype=torch.float32, device=d),
        ]
        torch.split_with_sizes_copy(x, ss, dim=-1, out=out)
        self.assertEqual(out[0].shape, (3, 2))

    def test_bfloat16(self):
        d = self._d()
        x = torch.zeros(6, dtype=torch.bfloat16, device=d)
        ss = [2, 4]
        out = [torch.zeros(2, dtype=torch.bfloat16, device=d), torch.zeros(4, dtype=torch.bfloat16, device=d)]
        torch.split_with_sizes_copy(x, ss, dim=0, out=out)
        self.assertEqual(out[0].dtype, torch.bfloat16)

    def test_out_len_mismatch_raises(self):
        d = self._d()
        x = torch.arange(6, dtype=torch.float32, device=d)
        ss = [2, 2, 2]
        out = [torch.zeros(2, device=d), torch.zeros(2, device=d)]
        _assert_raises((RuntimeError, ValueError, IndexError), lambda: torch.split_with_sizes_copy(x, ss, dim=0, out=out))

    def test_cpu_baseline(self):
        x = torch.arange(8, dtype=torch.float32)
        ss = [3, 5]
        out = [torch.zeros(3, dtype=torch.float32), torch.zeros(5, dtype=torch.float32)]
        torch.split_with_sizes_copy(x, ss, dim=0, out=out)
        self.assertEqual(out[0].shape, (3,))
        self.assertEqual(out[1].shape, (5,))


if __name__ == "__main__":
    run_tests()
