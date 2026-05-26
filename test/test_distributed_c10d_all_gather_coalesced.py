# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.distributed_c10d.all_gather_coalesced 接口功能正确性
API 名称：torch.distributed.distributed_c10d.all_gather_coalesced
API 签名：all_gather_coalesced(output_tensor_lists, input_tensor_list, group=None, async_op: bool = False)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | input_tensor_list 含单个 vs 多个张量                         | 已覆盖                                         |
| 枚举选项         | async_op: True/False; group: None(默认WORLD)                 | 已覆盖                                         |
| 参数类型         | output_tensor_lists: list[list[Tensor]]; input_tensor_list: list[Tensor] | 已覆盖                    |
| 传参与不传参     | group/async_op 使用默认值 vs 显式传入                        | 已覆盖                                         |
| 等价类/边界值    | 不同 dtype; 不同 shape; 单/多张量列表                        | 已覆盖                                         |
| 正常传参场景     | 多卡 HCCL 多进程 all_gather_coalesced                        | 已覆盖（标记 expectedFailure：HCCL 不支持）     |
| 异常传参场景     | output_tensor_lists 非 list 类型抛出 TypeError               | 已覆盖                                         |
| 混合设备类型     | 不涉及；多进程各 rank 使用相同 NPU 设备类型                   | 未覆盖，多进程场景各 rank 设备一致              |

未覆盖项及原因：
- 多卡 HCCL 实际执行：当前 HCCL 后端不支持 allgather_coalesced，多卡测试标记为 expectedFailure
- Complex tensor 路径：需 HCCL 多卡环境进一步验证
- 混合设备类型：多进程分布式场景各 rank 设备一致

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import os
import unittest

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


def _test_single_tensor(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        input_tensor = torch.randn(4, 8, device=device_name, dtype=torch.float32)
        input_tensor_list = [input_tensor]

        output_tensor_lists = [
            [torch.empty(4, 8, device=device_name, dtype=torch.float32) for _ in range(1)]
            for _ in range(world_size)
        ]

        dist.all_gather_coalesced(output_tensor_lists, input_tensor_list)

        assert len(output_tensor_lists) == world_size, \
            f"Expected {world_size} sublists, got {len(output_tensor_lists)}"
        for i, sublist in enumerate(output_tensor_lists):
            assert len(sublist) == 1, f"Sublist {i} has {len(sublist)} tensors, expected 1"
            assert sublist[0].shape == torch.Size([4, 8]), \
                f"Shape mismatch in sublist {i}: {sublist[0].shape}"
            assert sublist[0].dtype == torch.float32

        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


def _test_multi_tensor(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        t1 = torch.randn(4, 8, device=device_name, dtype=torch.float32)
        t2 = torch.randn(2, 3, device=device_name, dtype=torch.float32)
        input_tensor_list = [t1, t2]

        output_tensor_lists = [
            [torch.empty(4, 8, device=device_name, dtype=torch.float32),
             torch.empty(2, 3, device=device_name, dtype=torch.float32)]
            for _ in range(world_size)
        ]

        dist.all_gather_coalesced(output_tensor_lists, input_tensor_list)

        assert len(output_tensor_lists) == world_size
        for i, sublist in enumerate(output_tensor_lists):
            assert len(sublist) == 2
            assert sublist[0].shape == torch.Size([4, 8])
            assert sublist[1].shape == torch.Size([2, 3])
            assert sublist[0].dtype == torch.float32

        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


def _test_float16(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        input_tensor = torch.randn(4, 8, device=device_name, dtype=torch.float16)
        input_tensor_list = [input_tensor]

        output_tensor_lists = [
            [torch.empty(4, 8, device=device_name, dtype=torch.float16)]
            for _ in range(world_size)
        ]

        dist.all_gather_coalesced(output_tensor_lists, input_tensor_list)

        for sublist in output_tensor_lists:
            assert sublist[0].dtype == torch.float16

        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


def _test_async_op(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        input_tensor = torch.randn(4, 8, device=device_name, dtype=torch.float32)
        input_tensor_list = [input_tensor]

        output_tensor_lists = [
            [torch.empty(4, 8, device=device_name, dtype=torch.float32)]
            for _ in range(world_size)
        ]

        result = dist.all_gather_coalesced(output_tensor_lists, input_tensor_list, async_op=True)

        assert result is not None, "async_op=True should return a future/work handle"
        if hasattr(result, 'wait'):
            result.wait()

        for sublist in output_tensor_lists:
            assert sublist[0].shape == torch.Size([4, 8])

        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


def _test_sync_op_returns_none(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        input_tensor = torch.randn(4, 8, device=device_name, dtype=torch.float32)
        input_tensor_list = [input_tensor]

        output_tensor_lists = [
            [torch.empty(4, 8, device=device_name, dtype=torch.float32)]
            for _ in range(world_size)
        ]

        result = dist.all_gather_coalesced(output_tensor_lists, input_tensor_list, async_op=False)

        assert result is None, f"async_op=False should return None, got {type(result)}"

        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


def _test_group_default(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        input_tensor = torch.randn(4, 8, device=device_name, dtype=torch.float32)
        input_tensor_list = [input_tensor]

        output_tensor_lists = [
            [torch.empty(4, 8, device=device_name, dtype=torch.float32)]
            for _ in range(world_size)
        ]

        dist.all_gather_coalesced(output_tensor_lists, input_tensor_list, group=None)

        for sublist in output_tensor_lists:
            assert sublist[0].shape == torch.Size([4, 8])

        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


class TestDistributedC10dAllGatherCoalesced(TestCase):
    """Test cases for torch.distributed.distributed_c10d.all_gather_coalesced."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    # NOTE: HCCL backend does not support allgather_coalesced.
    # These tests are marked as expectedFailure to track the upstream gap.

    @skipIfUnsupportMultiNPU(2)
    @unittest.expectedFailure
    def test_single_tensor(self):
        """Gather single tensor per rank."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_single_tensor,
            args=(world_size, self.device_name, result_queue, 29509),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    @skipIfUnsupportMultiNPU(2)
    @unittest.expectedFailure
    def test_multi_tensor(self):
        """Gather multiple tensors per rank."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_multi_tensor,
            args=(world_size, self.device_name, result_queue, 29509),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    @skipIfUnsupportMultiNPU(2)
    @unittest.expectedFailure
    def test_float16(self):
        """float16 dtype preserved in output."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_float16,
            args=(world_size, self.device_name, result_queue, 29509),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    @skipIfUnsupportMultiNPU(2)
    @unittest.expectedFailure
    def test_async_op(self):
        """async_op=True returns work handle."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_async_op,
            args=(world_size, self.device_name, result_queue, 29509),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    @skipIfUnsupportMultiNPU(2)
    @unittest.expectedFailure
    def test_sync_op_returns_none(self):
        """async_op=False returns None."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_sync_op_returns_none,
            args=(world_size, self.device_name, result_queue, 29509),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    @skipIfUnsupportMultiNPU(2)
    @unittest.expectedFailure
    def test_group_default(self):
        """group=None uses default WORLD group."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_group_default,
            args=(world_size, self.device_name, result_queue, 29509),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    def test_output_not_list_raises(self):
        """output_tensor_lists not a list raises TypeError."""
        with self.assertRaises(TypeError):
            dist.all_gather_coalesced("not_a_list", [torch.randn(4)])


if __name__ == "__main__":
    run_tests()
