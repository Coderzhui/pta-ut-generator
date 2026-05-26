# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._functional_collectives.reduce_scatter_tensor_inplace 接口功能正确性
API 名称：torch.distributed._functional_collectives.reduce_scatter_tensor_inplace
API 签名：reduce_scatter_tensor_inplace(output: Tensor, input: Tensor, op: str = "sum", group=None, async_op: bool = False, scatter_dim: int = 0, tag: str = "")

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | tag 为空串 vs 非空串                                         | 已覆盖                                         |
| 枚举选项         | op: "sum", "avg"; scatter_dim: 0                             | 已覆盖                                         |
| 参数类型         | output/input: Tensor; op: str; scatter_dim: int              | 已覆盖                                         |
| 传参与不传参     | op/scatter_dim/tag 使用默认值 vs 显式传入                    | 已覆盖                                         |
| 等价类/边界值    | 1D/2D 张量；不同 dtype                                       | 已覆盖                                         |
| 正常传参场景     | 多卡 HCCL 多进程 reduce_scatter                              | 已覆盖                                         |
| 异常传参场景     | async_op=True 抛出 AssertionError                            | 已覆盖                                         |
| 混合设备类型     | 不涉及；多进程各 rank 使用相同 NPU 设备类型                   | 未覆盖，多进程场景各 rank 设备一致              |

未覆盖项及原因：
- op 非 sum/avg 值（product/min/max）：需 HCCL 多卡实际验证
- scatter_dim 非 0 值：需多卡环境验证
- 混合设备类型：多进程场景各 rank 设备一致

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import os
import random
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


def _test_basic_reduce_scatter(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        from torch.distributed._functional_collectives import reduce_scatter_tensor_inplace

        input_tensor = torch.randn(world_size * 4, 8, device=device_name, dtype=torch.float32)
        output_tensor = torch.empty(4, 8, device=device_name, dtype=torch.float32)
        reduce_scatter_tensor_inplace(output_tensor, input_tensor, group=dist.group.WORLD)

        assert output_tensor.shape == torch.Size([4, 8]), f"Shape mismatch: {output_tensor.shape}"
        assert output_tensor.dtype == torch.float32
        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


def _test_op_avg(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        from torch.distributed._functional_collectives import reduce_scatter_tensor_inplace

        input_tensor = torch.randn(world_size * 4, 8, device=device_name, dtype=torch.float32)
        output_tensor = torch.empty(4, 8, device=device_name, dtype=torch.float32)
        reduce_scatter_tensor_inplace(output_tensor, input_tensor, op="avg", group=dist.group.WORLD)

        assert output_tensor.shape == torch.Size([4, 8])
        assert output_tensor.dtype == torch.float32
        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


def _test_float16(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        from torch.distributed._functional_collectives import reduce_scatter_tensor_inplace

        input_tensor = torch.randn(world_size * 4, 8, device=device_name, dtype=torch.float16)
        output_tensor = torch.empty(4, 8, device=device_name, dtype=torch.float16)
        reduce_scatter_tensor_inplace(output_tensor, input_tensor, group=dist.group.WORLD)

        assert output_tensor.shape == torch.Size([4, 8])
        assert output_tensor.dtype == torch.float16
        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


def _test_1d_tensor(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        from torch.distributed._functional_collectives import reduce_scatter_tensor_inplace

        input_tensor = torch.randn(world_size * 8, device=device_name, dtype=torch.float32)
        output_tensor = torch.empty(8, device=device_name, dtype=torch.float32)
        reduce_scatter_tensor_inplace(output_tensor, input_tensor, group=dist.group.WORLD)

        assert output_tensor.shape == torch.Size([8])
        assert output_tensor.dtype == torch.float32
        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


def _test_custom_tag(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        from torch.distributed._functional_collectives import reduce_scatter_tensor_inplace

        input_tensor = torch.randn(world_size * 4, 8, device=device_name, dtype=torch.float32)
        output_tensor = torch.empty(4, 8, device=device_name, dtype=torch.float32)
        reduce_scatter_tensor_inplace(output_tensor, input_tensor, group=dist.group.WORLD, tag="scatter_tag")

        assert output_tensor.shape == torch.Size([4, 8])
        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


def _test_inplace_modification(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        from torch.distributed._functional_collectives import reduce_scatter_tensor_inplace

        input_tensor = torch.randn(world_size * 4, 8, device=device_name, dtype=torch.float32)
        output_tensor = torch.empty(4, 8, device=device_name, dtype=torch.float32)
        original_data_ptr = output_tensor.data_ptr()
        reduce_scatter_tensor_inplace(output_tensor, input_tensor, group=dist.group.WORLD)

        assert output_tensor.data_ptr() == original_data_ptr, "Output replaced instead of in-place modify"
        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


class TestReduceScatterTensorInplace(TestCase):
    """Test cases for torch.distributed._functional_collectives.reduce_scatter_tensor_inplace."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_basic_reduce_scatter(self):
        """Basic reduce_scatter with op=sum, scatter_dim=0."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_basic_reduce_scatter,
            args=(world_size, self.device_name, result_queue, random.randint(30000, 60000)),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    @skipIfUnsupportMultiNPU(2)
    def test_op_avg(self):
        """reduce_scatter with op=avg."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_op_avg,
            args=(world_size, self.device_name, result_queue, random.randint(30000, 60000)),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    @skipIfUnsupportMultiNPU(2)
    def test_float16(self):
        """float16 dtype preserved."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_float16,
            args=(world_size, self.device_name, result_queue, random.randint(30000, 60000)),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    @skipIfUnsupportMultiNPU(2)
    def test_1d_tensor(self):
        """1D tensor reduce_scatter."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_1d_tensor,
            args=(world_size, self.device_name, result_queue, random.randint(30000, 60000)),
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
            args=(world_size, self.device_name, result_queue, random.randint(30000, 60000)),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    @skipIfUnsupportMultiNPU(2)
    def test_inplace_modification(self):
        """Verify output modified in-place (same data_ptr)."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_inplace_modification,
            args=(world_size, self.device_name, result_queue, random.randint(30000, 60000)),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    def test_async_op_raises(self):
        """async_op=True raises AssertionError."""
        from torch.distributed._functional_collectives import reduce_scatter_tensor_inplace

        output = torch.empty(4, 8)
        input_t = torch.randn(8, 8)
        with self.assertRaises(AssertionError):
            reduce_scatter_tensor_inplace(output, input_t, async_op=True)


if __name__ == "__main__":
    run_tests()
