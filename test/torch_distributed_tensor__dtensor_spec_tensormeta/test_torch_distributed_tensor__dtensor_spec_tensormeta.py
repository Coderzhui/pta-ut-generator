# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.tensor._dtensor_spec.TensorMeta 命名元组
API 名称：torch.distributed.tensor._dtensor_spec.TensorMeta
API 签名：TensorMeta(shape: Size, stride: tuple[int, ...], dtype: dtype)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | N/A                                                          | N/A                                            |
| 枚举选项         | 多种 dtype                                                   | 已覆盖 float32 / bfloat16                      |
| 参数类型         | Size, tuple, dtype                                           | 已覆盖                                         |
| 传参与不传参     | N/A                                                          | N/A                                            |
| 等价类/边界值    | 不同 shape / stride                                          | 已覆盖                                         |
| 正常传参场景     | 字段可访问、NamedTuple 行为                                  | 已覆盖                                         |
| 异常传参场景     | N/A                                                          | 未覆盖                                         |

未覆盖项及原因：
- 无

注意：仅元数据构造与字段一致性，无数值计算。
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

from torch.distributed.tensor._dtensor_spec import TensorMeta


class TestTensorMeta(TestCase):
    def test_fields_float32(self):
        tm = TensorMeta(torch.Size([2, 4]), (4, 1), torch.float32)
        self.assertEqual(tm.shape, torch.Size([2, 4]))
        self.assertEqual(tm.stride, (4, 1))
        self.assertEqual(tm.dtype, torch.float32)

    def test_fields_bfloat16(self):
        tm = TensorMeta(torch.Size([1]), (1,), torch.bfloat16)
        self.assertEqual(tm.dtype, torch.bfloat16)

    def test_namedtuple_index(self):
        tm = TensorMeta(torch.Size([3, 3]), (3, 1), torch.float32)
        self.assertEqual(tm[0], torch.Size([3, 3]))
        self.assertEqual(tm[2], torch.float32)


if __name__ == "__main__":
    run_tests()
