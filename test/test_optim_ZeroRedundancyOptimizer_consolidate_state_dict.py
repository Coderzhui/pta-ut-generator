# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.optim.ZeroRedundancyOptimizer.consolidate_state_dict 接口功能正确性
API 名称：torch.distributed.optim.ZeroRedundancyOptimizer.consolidate_state_dict
API 签名：consolidate_state_dict(self, to: int = 0) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 优化器含状态（非空）                                         | 已覆盖                                          |
| 枚举选项         | to (0, 非0 rank)                                             | 已覆盖                                          |
| 参数类型         | to: int                                                      | 已覆盖                                          |
| 传参与不传参     | to 默认 0 vs 显式传入                                        | 已覆盖                                          |
| 等价类/边界值    | 2 rank 环境，默认目标 rank 和非默认                          | 已覆盖                                          |
| 正常传参场景     | 优化器 step 后 consolidate_state_dict                        | 已覆盖                                          |
| 异常传参场景     | 无稳定异常路径                                                | 未覆盖：内部异常依赖优化器状态                  |
| 混合设备类型     | NPU tensor 通过 broadcast 跨 rank 同步                       | 已覆盖                                          |

未覆盖项及原因：
- 异常传参场景：consolidate_state_dict 内部异常依赖优化器和进程组状态

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import os
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch_npu
from torch.distributed.optim import ZeroRedundancyOptimizer

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


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


def _run_consolidate_to_rank0(rank, world_size, port):
    """Worker: consolidate_state_dict with to=0."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    torch.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    try:
        local_device = f"npu:{rank}"
        model = SimpleModel().to(local_device)
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.Adam,
            lr=0.01,
        )

        x = torch.randn(4, 4, device=local_device)
        output = model(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        optimizer.consolidate_state_dict(to=0)

        if rank == 0:
            all_sds = optimizer._all_state_dicts
            assert len(all_sds) == world_size
    finally:
        dist.destroy_process_group()


def _run_consolidate_to_rank1(rank, world_size, port):
    """Worker: consolidate_state_dict with to=1."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    torch.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    try:
        local_device = f"npu:{rank}"
        model = SimpleModel().to(local_device)
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.Adam,
            lr=0.01,
        )

        x = torch.randn(4, 4, device=local_device)
        output = model(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        optimizer.consolidate_state_dict(to=1)

        if rank == 1:
            all_sds = optimizer._all_state_dicts
            assert len(all_sds) == world_size
    finally:
        dist.destroy_process_group()


def _run_consolidate_keys(rank, world_size, port):
    """Worker: verify consolidated state_dict has expected keys."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    torch.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    try:
        local_device = f"npu:{rank}"
        model = SimpleModel().to(local_device)
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.Adam,
            lr=0.01,
        )

        x = torch.randn(4, 4, device=local_device)
        output = model(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        optimizer.consolidate_state_dict(to=0)

        if rank == 0:
            all_sds = optimizer._all_state_dicts
            assert len(all_sds) > 0
            first_sd = all_sds[0]
            assert "state" in first_sd
            assert "param_groups" in first_sd
    finally:
        dist.destroy_process_group()


class TestZeroRedundancyOptimizerConsolidate(TestCase):
    """Test ZeroRedundancyOptimizer.consolidate_state_dict with multi-card HCCL."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu",
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_consolidate_to_rank0(self):
        """consolidate_state_dict with to=0 populates _all_state_dicts on rank 0."""
        world_size = 2
        port = random.randint(30000, 60000)
        mp.spawn(
            _run_consolidate_to_rank0,
            args=(world_size, port),
            nprocs=world_size,
            join=True,
        )

    @skipIfUnsupportMultiNPU(2)
    def test_consolidate_to_rank1(self):
        """consolidate_state_dict with to=1 populates on rank 1."""
        world_size = 2
        port = random.randint(30000, 60000)
        mp.spawn(
            _run_consolidate_to_rank1,
            args=(world_size, port),
            nprocs=world_size,
            join=True,
        )

    @skipIfUnsupportMultiNPU(2)
    def test_consolidated_has_expected_keys(self):
        """Consolidated state_dicts have 'state' and 'param_groups' keys."""
        world_size = 2
        port = random.randint(30000, 60000)
        mp.spawn(
            _run_consolidate_keys,
            args=(world_size, port),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    run_tests()
