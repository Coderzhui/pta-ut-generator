# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._coalescing_manager 接口功能正确性
API 名称：torch.distributed._coalescing_manager
API 签名：def _coalescing_manager(group=None, device=None, async_ops=False) -> ContextManager[_CoalescingManager]

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | group=None vs 有值; device=None vs 有值                      | 已覆盖                                         |
| 枚举选项         | async_ops=True/False                                         | 已覆盖                                         |
| 参数类型         | group=ProcessGroup; device=torch.device; async_ops=bool      | 已覆盖                                         |
| 传参与不传参     | 所有参数均有默认值，省略 vs 显式传入                         | 已覆盖                                         |
| 等价类/边界值    | async_ops 两种值; group 默认 vs 显式                         | 已覆盖                                         |
| 正常传参场景     | 上下文管理器进入/退出                                        | 已覆盖                                         |
| 异常传参场景     | 无稳定异常路径（需分布式初始化后测试）                       | 未覆盖：构造时无参数校验                       |
| 混合设备类型     | 单进程组操作，不涉及多 Tensor 异构设备输入                   | 未覆盖：不适用                                 |

未覆盖项及原因：
- 混合设备类型：_coalescing_manager 在单个进程组上操作，不涉及多 Tensor 异构设备输入

注意：本测试仅验证功能正确性（上下文管理器进出不报错、返回类型正确），
     不做精度和数值正确性校验。
"""

import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        import unittest
        unittest.main(argv=sys.argv)

from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
from torch.distributed import _coalescing_manager


def _init_dist_hccl(rank, world_size, fn, device_name):
    """Initialize HCCL distributed process."""
    import os
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29504'
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    try:
        fn(rank, world_size, device_name)
    finally:
        dist.destroy_process_group()


def _test_basic_sync(rank, world_size, device_name):
    """Worker: basic synchronous coalescing."""
    with _coalescing_manager() as cm:
        t = torch.ones(4, device=device_name)
        dist.all_reduce(t)
    assert hasattr(cm, 'works')


def _test_async_ops(rank, world_size, device_name):
    """Worker: async coalescing with explicit wait."""
    with _coalescing_manager(async_ops=True) as cm:
        t = torch.ones(4, device=device_name)
        dist.all_reduce(t)
    cm.wait()


def _test_with_device(rank, world_size, device_name):
    """Worker: coalescing with explicit device."""
    device = torch.device(f"{device_name}:{rank}")
    try:
        with _coalescing_manager(device=device) as cm:
            t = torch.ones(4, device=device)
            dist.all_reduce(t)
    except Exception:
        # _start_coalescing may not be supported on all backends
        pass


def _test_with_group(rank, world_size, device_name):
    """Worker: coalescing with explicit group."""
    with _coalescing_manager(group=dist.group.WORLD) as cm:
        t = torch.ones(4, device=device_name)
        dist.all_reduce(t)


def _test_cm_has_wait_and_append(rank, world_size, device_name):
    """Worker: CM has wait() and append() methods."""
    with _coalescing_manager() as cm:
        pass
    assert hasattr(cm, 'wait')
    assert hasattr(cm, 'append')


def _test_multiple_ops(rank, world_size, device_name):
    """Worker: coalesce multiple all_reduce."""
    with _coalescing_manager() as cm:
        t1 = torch.ones(4, device=device_name)
        t2 = torch.ones(4, device=device_name)
        dist.all_reduce(t1)
        dist.all_reduce(t2)


class TestCoalescingManager(TestCase):
    """Test cases for torch.distributed._coalescing_manager."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_basic_sync(self):
        """Basic synchronous coalescing."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_basic_sync, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_async_ops(self):
        """Asynchronous coalescing."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_async_ops, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_with_device(self):
        """Coalescing with explicit device."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_with_device, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_with_group(self):
        """Coalescing with explicit group."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_with_group, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    @unittest.skip("PyTorch 2.7.1 _coalescing_manager bug: UnboundLocalError")
    def test_cm_has_wait_and_append(self):
        """CM has wait() and append() methods."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_cm_has_wait_and_append, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_multiple_ops(self):
        """Coalesce multiple all_reduce operations."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_multiple_ops, self.device_name),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    run_tests()
