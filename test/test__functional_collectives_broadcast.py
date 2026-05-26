# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._functional_collectives.broadcast 接口功能正确性
API 名称：torch.distributed._functional_collectives.broadcast
API 签名：broadcast(self: Tensor, src: int, group: RANK_TYPES, tag: str = "")

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | tag 为空串 vs 非空串                                         | 已覆盖                                         |
| 枚举选项         | src: 0, 1; tag: "" vs "custom_tag"                           | 已覆盖                                         |
| 参数类型         | self: Tensor; src: int; group: ProcessGroup; tag: str        | 已覆盖                                         |
| 传参与不传参     | tag 默认空串 vs 显式传入                                     | 已覆盖                                         |
| 等价类/边界值    | 1D/2D 张量；不同 dtype；标量张量                             | 已覆盖                                         |
| 正常传参场景     | 多卡 HCCL 多进程从 src=0 和 src=1 广播                      | 已覆盖                                         |
| 异常传参场景     | 无稳定异常路径（函数内部不抛出可预期异常）                   | 未覆盖，函数无显式异常分支                      |
| 混合设备类型     | 不涉及；多进程各 rank 使用相同 NPU 设备类型                   | 未覆盖，多进程场景各 rank 设备一致              |

未覆盖项及原因：
- 异常传参场景：函数无显式异常路径，无法稳定触发
- 混合设备类型：多进程分布式场景各 rank 设备一致

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


def _test_broadcast_from_rank0(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        from torch.distributed._functional_collectives import broadcast

        tensor = torch.randn(4, 8, device=device_name, dtype=torch.float32)
        result = broadcast(tensor, src=0, group=dist.group.WORLD)

        assert result.shape == torch.Size([4, 8]), f"Shape mismatch: {result.shape}"
        assert result.dtype == torch.float32, f"Dtype mismatch: {result.dtype}"
        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


def _test_broadcast_from_rank1(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        from torch.distributed._functional_collectives import broadcast

        tensor = torch.randn(4, 8, device=device_name, dtype=torch.float32)
        result = broadcast(tensor, src=1, group=dist.group.WORLD)

        assert result.shape == torch.Size([4, 8]), f"Shape mismatch: {result.shape}"
        assert result.dtype == torch.float32, f"Dtype mismatch: {result.dtype}"
        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


def _test_broadcast_1d(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        from torch.distributed._functional_collectives import broadcast

        tensor = torch.randn(16, device=device_name, dtype=torch.float32)
        result = broadcast(tensor, src=0, group=dist.group.WORLD)

        assert result.shape == torch.Size([16]), f"Shape mismatch: {result.shape}"
        assert result.dtype == torch.float32
        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


def _test_broadcast_float16(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        from torch.distributed._functional_collectives import broadcast

        tensor = torch.randn(4, 8, device=device_name, dtype=torch.float16)
        result = broadcast(tensor, src=0, group=dist.group.WORLD)

        assert result.shape == torch.Size([4, 8])
        assert result.dtype == torch.float16
        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


def _test_broadcast_int32(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        from torch.distributed._functional_collectives import broadcast

        tensor = torch.randint(0, 100, (4, 8), device=device_name, dtype=torch.int32)
        result = broadcast(tensor, src=0, group=dist.group.WORLD)

        assert result.shape == torch.Size([4, 8])
        assert result.dtype == torch.int32
        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


def _test_broadcast_custom_tag(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        from torch.distributed._functional_collectives import broadcast

        tensor = torch.randn(4, 8, device=device_name, dtype=torch.float32)
        result = broadcast(tensor, src=0, group=dist.group.WORLD, tag="test_broadcast_tag")

        assert result.shape == torch.Size([4, 8])
        assert result.dtype == torch.float32
        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


def _test_broadcast_scalar(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        from torch.distributed._functional_collectives import broadcast

        tensor = torch.tensor(3.14, device=device_name, dtype=torch.float32)
        result = broadcast(tensor, src=0, group=dist.group.WORLD)

        assert result.shape == torch.Size([])
        assert result.dtype == torch.float32
        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


class TestFunctionalCollectivesBroadcast(TestCase):
    """Test cases for torch.distributed._functional_collectives.broadcast."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_broadcast_from_rank0(self):
        """Broadcast 2D tensor from rank 0."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_broadcast_from_rank0,
            args=(world_size, self.device_name, result_queue, 29501),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    @skipIfUnsupportMultiNPU(2)
    def test_broadcast_from_rank1(self):
        """Broadcast 2D tensor from rank 1."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_broadcast_from_rank1,
            args=(world_size, self.device_name, result_queue, 29501),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    @skipIfUnsupportMultiNPU(2)
    def test_broadcast_1d_tensor(self):
        """Broadcast 1D tensor."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_broadcast_1d,
            args=(world_size, self.device_name, result_queue, 29501),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    @skipIfUnsupportMultiNPU(2)
    def test_broadcast_float16(self):
        """Broadcast with float16 dtype preserved."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_broadcast_float16,
            args=(world_size, self.device_name, result_queue, 29501),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    @skipIfUnsupportMultiNPU(2)
    def test_broadcast_int32(self):
        """Broadcast with int32 dtype preserved."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_broadcast_int32,
            args=(world_size, self.device_name, result_queue, 29501),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    @skipIfUnsupportMultiNPU(2)
    def test_broadcast_custom_tag(self):
        """Broadcast with custom tag string."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_broadcast_custom_tag,
            args=(world_size, self.device_name, result_queue, 29501),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    @skipIfUnsupportMultiNPU(2)
    def test_broadcast_scalar(self):
        """Broadcast scalar (0-dim) tensor."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_broadcast_scalar,
            args=(world_size, self.device_name, result_queue, 29501),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")


if __name__ == "__main__":
    run_tests()
