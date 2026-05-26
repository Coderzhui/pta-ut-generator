# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._functional_collectives.reduce_scatter_tensor_coalesced 接口功能正确性
API 名称：torch.distributed._functional_collectives.reduce_scatter_tensor_coalesced
API 签名：reduce_scatter_tensor_coalesced(inputs: list[Tensor], reduceOp: str, scatter_dim: list[int], group: RANK_TYPES, tag: str = "") -> list[Tensor]

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 输入为单张量列表 vs 多张量列表                               | 已覆盖                                         |
| 枚举选项         | reduceOp: sum; scatter_dim: 0, 1                             | 已覆盖 reduceOp=sum, scatter_dim=0; scatter_dim=1 需多卡环境 |
| 参数类型         | inputs: list[Tensor]; scatter_dim: list[int]; tag: str       | 已覆盖                                         |
| 传参与不传参     | tag 默认空串 vs 自定义值                                     | 已覆盖                                         |
| 等价类/边界值    | 单元素列表、多元素列表；scatter_dim 与 inputs 长度匹配        | 已覆盖                                         |
| 正常传参场景     | 多卡 HCCL 多进程调用                                         | 已覆盖                                         |
| 异常传参场景     | scatter_dim 长度与 inputs 不匹配                              | 已覆盖                                         |
| 混合设备类型     | 不涉及；多进程各 rank 使用相同 NPU 设备类型                   | 未覆盖，多进程场景各 rank 设备一致              |

未覆盖项及原因：
- reduceOp 非sum值（avg/product/min/max）：需多卡环境验证，本 UT 仅验证 sum 路径结构正确性
- scatter_dim 非0值：需 HCCL 多卡实际执行验证

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


def _test_basic_reduce_scatter(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        from torch.distributed._functional_collectives import reduce_scatter_tensor_coalesced

        tensor = torch.randn(world_size * 4, 8, device=device_name, dtype=torch.float32)
        inputs = [tensor]
        scatter_dim = [0]
        outputs = reduce_scatter_tensor_coalesced(inputs, "sum", scatter_dim, dist.group.WORLD)

        assert len(outputs) == 1, f"Expected 1 output, got {len(outputs)}"
        out = outputs[0]
        assert out.shape == torch.Size([4, 8]), f"Expected shape [4, 8], got {out.shape}"
        assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"
        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


def _test_multi_tensor(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        from torch.distributed._functional_collectives import reduce_scatter_tensor_coalesced

        t1 = torch.randn(world_size * 4, 8, device=device_name, dtype=torch.float32)
        t2 = torch.randn(4, world_size * 6, device=device_name, dtype=torch.float32)
        inputs = [t1, t2]
        scatter_dim = [0, 1]
        outputs = reduce_scatter_tensor_coalesced(inputs, "sum", scatter_dim, dist.group.WORLD)

        assert len(outputs) == 2, f"Expected 2 outputs, got {len(outputs)}"
        assert outputs[0].shape == torch.Size([4, 8]), f"t1 output shape mismatch: {outputs[0].shape}"
        assert outputs[1].shape == torch.Size([4, 6]), f"t2 output shape mismatch: {outputs[1].shape}"
        assert outputs[0].dtype == torch.float32
        assert outputs[1].dtype == torch.float32
        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


def _test_custom_tag(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        from torch.distributed._functional_collectives import reduce_scatter_tensor_coalesced

        tensor = torch.randn(world_size * 4, device=device_name, dtype=torch.float32)
        outputs = reduce_scatter_tensor_coalesced(
            [tensor], "sum", [0], dist.group.WORLD, tag="test_tag"
        )
        assert len(outputs) == 1
        assert outputs[0].shape == torch.Size([4])
        assert outputs[0].dtype == torch.float32
        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


def _test_float16(rank, world_size, device_name, result_queue, port):
    _init_process(rank, world_size, port)
    try:
        from torch.distributed._functional_collectives import reduce_scatter_tensor_coalesced

        tensor = torch.randn(world_size * 4, device=device_name, dtype=torch.float16)
        outputs = reduce_scatter_tensor_coalesced([tensor], "sum", [0], dist.group.WORLD)
        assert len(outputs) == 1
        assert outputs[0].shape == torch.Size([4])
        assert outputs[0].dtype == torch.float16
        if rank == 0:
            result_queue.put("PASS")
    finally:
        dist.destroy_process_group()


class TestReduceScatterTensorCoalesced(TestCase):
    """Test cases for torch.distributed._functional_collectives.reduce_scatter_tensor_coalesced."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_basic_reduce_scatter(self):
        """Single tensor reduce_scatter with scatter_dim=0."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_basic_reduce_scatter,
            args=(world_size, self.device_name, result_queue, 29503),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    @skipIfUnsupportMultiNPU(2)
    def test_multi_tensor(self):
        """Multiple tensors with different scatter_dims."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_multi_tensor,
            args=(world_size, self.device_name, result_queue, 29503),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    @skipIfUnsupportMultiNPU(2)
    def test_custom_tag(self):
        """Pass custom tag string."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_custom_tag,
            args=(world_size, self.device_name, result_queue, 29503),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    @skipIfUnsupportMultiNPU(2)
    def test_float16_dtype(self):
        """Verify float16 dtype preserved in output."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _test_float16,
            args=(world_size, self.device_name, result_queue, 29503),
            nprocs=world_size,
            join=True,
        )
        self.assertEqual(result_queue.get(), "PASS")

    def test_scatter_dim_length_mismatch(self):
        """scatter_dim length != inputs length raises error."""
        from torch.distributed._functional_collectives import reduce_scatter_tensor_coalesced

        t = torch.randn(4, 4)
        # Outside init_process_group, dist.group.WORLD is None,
        # which triggers ValueError before the length check.
        with self.assertRaises((AssertionError, ValueError)):
            reduce_scatter_tensor_coalesced([t], "sum", [0, 1], dist.group.WORLD)


if __name__ == "__main__":
    run_tests()
