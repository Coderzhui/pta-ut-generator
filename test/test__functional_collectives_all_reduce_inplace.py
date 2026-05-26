# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._functional_collectives.all_reduce_inplace 接口功能正确性
API 名称：torch.distributed._functional_collectives.all_reduce_inplace
API 签名：all_reduce_inplace(tensor: Tensor, op: str = "sum", group=None, async_op: bool = False, tag: str = "")

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | tag 为空串 vs 非空串                                         | 已覆盖                                         |
| 枚举选项         | op: "sum", "avg", "max", "min"                               | 已覆盖                                         |
| 参数类型         | tensor: Tensor; op: str; group: ProcessGroup; tag: str       | 已覆盖                                         |
| 传参与不传参     | op/tag 使用默认值 vs 显式传入                                | 已覆盖                                         |
| 等价类/边界值    | 1D/2D 张量；不同 dtype                                       | 已覆盖                                         |
| 正常传参场景     | 多卡 HCCL 多进程 all_reduce                                  | 已覆盖                                         |
| 异常传参场景     | async_op=True 抛出 AssertionError                            | 已覆盖                                         |
| 混合设备类型     | 不涉及；多进程各 rank 使用相同 NPU 设备类型                   | 未覆盖，多进程场景各 rank 设备一致              |

未覆盖项及原因：
- op 非 sum/avg/max/min 值（product/band/bor/bxor）：需 HCCL 验证
- 混合设备类型：多进程场景各 rank 设备一致

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import torch_npu  # noqa: F401
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


def _init_process(rank, world_size, port):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)


def _test_basic_all_reduce(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        from torch.distributed._functional_collectives import all_reduce_inplace

        tensor = torch.randn(4, 8, device=device_name, dtype=torch.float32)
        original_shape = tensor.shape
        original_dtype = tensor.dtype
        all_reduce_inplace(tensor, group=dist.group.WORLD)

        assert tensor.shape == original_shape, f"Shape changed: {tensor.shape} != {original_shape}"
        assert tensor.dtype == original_dtype, f"Dtype changed: {tensor.dtype} != {original_dtype}"
        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


def _test_op_avg(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        from torch.distributed._functional_collectives import all_reduce_inplace

        tensor = torch.randn(4, 8, device=device_name, dtype=torch.float32)
        all_reduce_inplace(tensor, op="avg", group=dist.group.WORLD)

        assert tensor.shape == torch.Size([4, 8])
        assert tensor.dtype == torch.float32
        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


def _test_op_max(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        from torch.distributed._functional_collectives import all_reduce_inplace

        tensor = torch.randn(4, 8, device=device_name, dtype=torch.float32)
        all_reduce_inplace(tensor, op="max", group=dist.group.WORLD)

        assert tensor.shape == torch.Size([4, 8])
        assert tensor.dtype == torch.float32
        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


def _test_op_min(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        from torch.distributed._functional_collectives import all_reduce_inplace

        tensor = torch.randn(4, 8, device=device_name, dtype=torch.float32)
        all_reduce_inplace(tensor, op="min", group=dist.group.WORLD)

        assert tensor.shape == torch.Size([4, 8])
        assert tensor.dtype == torch.float32
        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


def _test_float16(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        from torch.distributed._functional_collectives import all_reduce_inplace

        tensor = torch.randn(4, 8, device=device_name, dtype=torch.float16)
        all_reduce_inplace(tensor, group=dist.group.WORLD)

        assert tensor.shape == torch.Size([4, 8])
        assert tensor.dtype == torch.float16
        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


def _test_1d_tensor(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        from torch.distributed._functional_collectives import all_reduce_inplace

        tensor = torch.randn(16, device=device_name, dtype=torch.float32)
        all_reduce_inplace(tensor, group=dist.group.WORLD)

        assert tensor.shape == torch.Size([16])
        assert tensor.dtype == torch.float32
        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


def _test_custom_tag(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        from torch.distributed._functional_collectives import all_reduce_inplace

        tensor = torch.randn(4, 8, device=device_name, dtype=torch.float32)
        all_reduce_inplace(tensor, group=dist.group.WORLD, tag="reduce_tag")

        assert tensor.shape == torch.Size([4, 8])
        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


def _test_inplace_storage(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        from torch.distributed._functional_collectives import all_reduce_inplace

        tensor = torch.randn(4, 8, device=device_name, dtype=torch.float32)
        original_data_ptr = tensor.data_ptr()
        all_reduce_inplace(tensor, group=dist.group.WORLD)

        assert tensor.data_ptr() == original_data_ptr, "Tensor was replaced, not modified in-place"
        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


class TestAllReduceInplace(TestCase):
    """Test cases for torch.distributed._functional_collectives.all_reduce_inplace."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_basic_all_reduce(self):
        """Basic all_reduce with default op=sum."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_basic_all_reduce,
            args=(world_size, self.device_name, result_queue, 29506),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    @skipIfUnsupportMultiNPU(2)
    def test_op_avg(self):
        """all_reduce with op=avg."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_op_avg,
            args=(world_size, self.device_name, result_queue, 29506),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    @skipIfUnsupportMultiNPU(2)
    def test_op_max(self):
        """all_reduce with op=max."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_op_max,
            args=(world_size, self.device_name, result_queue, 29506),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    @skipIfUnsupportMultiNPU(2)
    def test_op_min(self):
        """all_reduce with op=min."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_op_min,
            args=(world_size, self.device_name, result_queue, 29506),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    @skipIfUnsupportMultiNPU(2)
    def test_float16(self):
        """float16 dtype preserved after all_reduce."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_float16,
            args=(world_size, self.device_name, result_queue, 29506),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    @skipIfUnsupportMultiNPU(2)
    def test_1d_tensor(self):
        """1D tensor all_reduce."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_1d_tensor,
            args=(world_size, self.device_name, result_queue, 29506),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    @skipIfUnsupportMultiNPU(2)
    def test_custom_tag(self):
        """Custom tag string."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_custom_tag,
            args=(world_size, self.device_name, result_queue, 29506),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    @skipIfUnsupportMultiNPU(2)
    def test_inplace_storage(self):
        """Verify tensor modified in-place (same data_ptr)."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_inplace_storage,
            args=(world_size, self.device_name, result_queue, 29506),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    def test_async_op_raises(self):
        """async_op=True raises AssertionError."""
        from torch.distributed._functional_collectives import all_reduce_inplace

        tensor = torch.randn(4, 8)
        with self.assertRaises(AssertionError):
            all_reduce_inplace(tensor, async_op=True)


if __name__ == "__main__":
    run_tests()
