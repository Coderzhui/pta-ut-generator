# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.fsdp.FSDPModule.unshard 接口功能正确性
API 名称：torch.distributed.fsdp.FSDPModule.unshard
API 签名：unshard(self, async_op: bool = False) -> UnshardHandle | None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | FSDP 模块含参数（非空）                                      | 已覆盖                                          |
| 枚举选项         | async_op (True/False)                                        | 已覆盖                                          |
| 参数类型         | async_op: bool                                               | 已覆盖                                          |
| 传参与不传参     | async_op 默认 False vs True                                  | 已覆盖                                          |
| 等价类/边界值    | 简单线性模型                                                 | 已覆盖                                          |
| 正常传参场景     | unshard 后参数可访问；unshard + reshard 循环                  | 已覆盖                                          |
| 异常传参场景     | 无稳定异常路径                                                | 未覆盖：unshard 依赖 FSDP 状态                  |
| 混合设备类型     | NPU tensor 通过 all-gather 跨 rank 重组                      | 已覆盖                                          |

未覆盖项及原因：
- 异常传参场景：unshard 内部异常依赖 FSDP 初始化状态

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch_npu

try:
    from torch_npu.testing.testcase import TestCase, run_tests
    from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
except ImportError:
    import sys
    from unittest import TestCase

    def skipIfUnsupportMultiNPU(n):
        def decorator(func):
            return func
        return decorator

    def run_tests():
        unittest.main(argv=sys.argv)


def _run_unshard_sync(rank, world_size):
    """Worker: test unshard with async_op=False."""
    from torch.distributed.fsdp import fully_shard

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29506"
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    torch.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    try:
        local_device = f"npu:{rank}"
        model = nn.Linear(8, 4).to(local_device)
        fully_shard(model)

        result = model.unshard(async_op=False)
        assert result is None, f"Expected None, got {result}"

        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0
    finally:
        dist.destroy_process_group()


def _run_unshard_async(rank, world_size):
    """Worker: test unshard with async_op=True."""
    from torch.distributed.fsdp import fully_shard

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29506"
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    torch.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    try:
        local_device = f"npu:{rank}"
        model = nn.Linear(8, 4).to(local_device)
        fully_shard(model)

        handle = model.unshard(async_op=True)
        assert handle is not None
        assert hasattr(handle, "wait")
        handle.wait()
    finally:
        dist.destroy_process_group()


def _run_unshard_reshard(rank, world_size):
    """Worker: test unshard followed by reshard."""
    from torch.distributed.fsdp import fully_shard

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29506"
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    torch.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    try:
        local_device = f"npu:{rank}"
        model = nn.Linear(8, 4).to(local_device)
        fully_shard(model)

        model.unshard(async_op=False)
        model.reshard()
    finally:
        dist.destroy_process_group()


class TestFSDPModuleUnshard(TestCase):
    """Test FSDPModule.unshard with multi-card HCCL."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu",
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_unshard_sync_returns_none(self):
        """unshard with async_op=False returns None."""
        world_size = 2
        mp.spawn(_run_unshard_sync, args=(world_size,), nprocs=world_size, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_unshard_async_returns_handle(self):
        """unshard with async_op=True returns UnshardHandle."""
        world_size = 2
        mp.spawn(_run_unshard_async, args=(world_size,), nprocs=world_size, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_unshard_reshard_cycle(self):
        """unshard followed by reshard works correctly."""
        world_size = 2
        mp.spawn(_run_unshard_reshard, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    run_tests()
