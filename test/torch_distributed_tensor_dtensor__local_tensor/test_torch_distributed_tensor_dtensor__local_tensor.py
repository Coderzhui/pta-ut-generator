# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.tensor.DTensor._local_tensor 与 from_local 一致性
API 名称：torch.distributed.tensor.DTensor._local_tensor
API 签名：实例属性，类型为 torch.Tensor

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | N/A                                                          | N/A                                            |
| 枚举选项         | Replicate 放置                                               | 已覆盖                                         |
| 参数类型         | local float Tensor                                           | 已覆盖                                         |
| 传参与不传参     | from_local run_check False / True（若可用）                  | 已覆盖 False；True 未强制                      |
| 等价类/边界值    | 不同 shape                                                   | 已覆盖                                         |
| 正常传参场景     | _local_tensor.shape/dtype/device 与传入 local 一致           | 已覆盖                                         |
| 异常传参场景     | N/A                                                          | 未覆盖                                         |

未覆盖项及原因：
- Shard/Partial 全矩阵：依赖多 rank 一致输入，单测仅 Replicate

注意：使用 init_device_mesh(cpu, (1,))；无数值断言。
"""

import unittest

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate

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


def _privateuse1_ok():
    name = torch._C._get_privateuse1_backend_name()
    m = getattr(torch, name, None)
    return name, m is not None and getattr(m, "is_available", lambda: False)()


class TestDTensorLocalTensor(TestCase):
    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_local_tensor_npu_mesh_float32(self):
        name, ok = _privateuse1_ok()
        if not ok:
            self.skipTest(f"{name} not available")
        mesh = init_device_mesh(name, (1,))
        try:
            lt = torch.randn(3, 3, dtype=torch.float32, device=torch.device(name, 0))
            dt = DTensor.from_local(lt, mesh, [Replicate()], run_check=False)
            self.assertEqual(dt._local_tensor.device.type, name)
            self.assertEqual(dt._local_tensor.shape, lt.shape)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    def test_local_tensor_npu_mesh_bfloat16(self):
        name, ok = _privateuse1_ok()
        if not ok:
            self.skipTest(f"{name} not available")
        mesh = init_device_mesh(name, (1,))
        try:
            lt = torch.ones(2, 4, dtype=torch.bfloat16, device=torch.device(name, 0))
            dt = DTensor.from_local(lt, mesh, [Replicate()], run_check=False)
            self.assertEqual(dt._local_tensor.dtype, torch.bfloat16)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    def test_local_tensor_npu_mesh_second_shape(self):
        name, ok = _privateuse1_ok()
        if not ok:
            self.skipTest(f"{name} not available")
        mesh = init_device_mesh(name, (1,))
        try:
            lt = torch.zeros(8, dtype=torch.float32, device=torch.device(name, 0))
            dt = DTensor.from_local(lt, mesh, [Replicate()], run_check=False)
            self.assertEqual(tuple(dt._local_tensor.shape), (8,))
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    def test_local_tensor_npu_1d(self):
        name, ok = _privateuse1_ok()
        if not ok:
            self.skipTest(f"{name} not available")
        mesh = init_device_mesh(name, (1,))
        try:
            lt = torch.zeros(16, dtype=torch.float32, device=torch.device(name, 0))
            dt = DTensor.from_local(lt, mesh, [Replicate()], run_check=False)
            self.assertEqual(dt._local_tensor.ndim, 1)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    def test_local_tensor_matches_from_local_cpu_mesh(self):
        mesh = init_device_mesh("cpu", (1,))
        lt = torch.randn(2, 5, dtype=torch.float32)
        dt = DTensor.from_local(lt, mesh, [Replicate()], run_check=False)
        self.assertIsInstance(dt._local_tensor, torch.Tensor)
        self.assertEqual(dt._local_tensor.shape, lt.shape)
        self.assertEqual(dt._local_tensor.dtype, lt.dtype)


if __name__ == "__main__":
    run_tests()
