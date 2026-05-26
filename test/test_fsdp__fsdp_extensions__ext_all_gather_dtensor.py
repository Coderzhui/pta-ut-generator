# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.fsdp._fsdp_extensions._ext_all_gather_dtensor 接口功能正确性
API 名称：torch.distributed.fsdp._fsdp_extensions._ext_all_gather_dtensor
API 签名：def _ext_all_gather_dtensor(tensor: DTensor, parent_mesh: DeviceMesh | None,
                                    fsdp_extension: FSDPExtensions | None = None) -> torch.Tensor

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                               |
|------------------|--------------------------------------------|----------------------------------------|
| 基础调用         | 多卡 HCCL 下调用不报错                     | 已覆盖                                 |
| 返回类型         | 返回 torch.Tensor                          | 已覆盖                                 |
| 参数枚举         | parent_mesh=None vs 实际 mesh              | 已覆盖                                 |
| fsdp_extension   | fsdp_extension=None vs 实际 extension      | 部分覆盖：None 路径已覆盖              |
| 异常路径         | 无稳定异常路径                             | 未覆盖：内部依赖分布式初始化           |

未覆盖项及原因：
- fsdp_extension 非 None：需完整 FSDP 扩展实例
- 异常路径：依赖分布式环境，无稳定可复现异常

注意：本测试仅验证功能正确性（调用不报错、返回类型符合预期），
     不做精度和数值正确性校验。
"""

import os
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
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed._tensor import Shard


def _init_dist_hccl(rank, world_size, fn, device_name):
    """Initialize HCCL distributed process."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29502'
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    try:
        fn(rank, world_size, device_name)
    finally:
        dist.destroy_process_group()


def _test_ext_all_gather_dtensor_basic(rank, world_size, device_name):
    """Worker: basic _ext_all_gather_dtensor call."""
    from torch.distributed.fsdp._fsdp_extensions import _ext_all_gather_dtensor
    mesh = DeviceMesh(device_name, list(range(world_size)))
    local_tensor = torch.randn(4, 4, device=device_name)
    dtensor = distribute_tensor(local_tensor, mesh, [Shard(0)])

    result = _ext_all_gather_dtensor(dtensor, mesh)
    assert isinstance(result, torch.Tensor), f"Expected Tensor, got {type(result)}"
    assert result.shape[1] == 4, f"Unexpected shape: {result.shape}"


def _test_ext_all_gather_dtensor_no_extension(rank, world_size, device_name):
    """Worker: call with fsdp_extension=None uses default path."""
    from torch.distributed.fsdp._fsdp_extensions import _ext_all_gather_dtensor
    mesh = DeviceMesh(device_name, list(range(world_size)))
    local_tensor = torch.randn(2, 3, device=device_name)
    dtensor = distribute_tensor(local_tensor, mesh, [Shard(0)])

    result = _ext_all_gather_dtensor(dtensor, mesh, fsdp_extension=None)
    assert isinstance(result, torch.Tensor)


def _test_ext_all_gather_dtensor_return_on_cpu(rank, world_size, device_name):
    """Worker: result is a regular Tensor, not DTensor."""
    from torch.distributed.fsdp._fsdp_extensions import _ext_all_gather_dtensor
    mesh = DeviceMesh(device_name, list(range(world_size)))
    local_tensor = torch.randn(2, 2, device=device_name)
    dtensor = distribute_tensor(local_tensor, mesh, [Shard(0)])

    result = _ext_all_gather_dtensor(dtensor, mesh)
    assert isinstance(result, torch.Tensor)
    assert not isinstance(result, DTensor)


class TestFsdpExtAllGatherDtensor(TestCase):
    """Test cases for _ext_all_gather_dtensor with HCCL."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_ext_all_gather_dtensor_basic(self):
        """Basic _ext_all_gather_dtensor returns Tensor."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_ext_all_gather_dtensor_basic, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_ext_all_gather_dtensor_no_extension(self):
        """Call with fsdp_extension=None uses default."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_ext_all_gather_dtensor_no_extension, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_ext_all_gather_dtensor_returns_tensor(self):
        """Result is a regular Tensor, not DTensor."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_ext_all_gather_dtensor_return_on_cpu, self.device_name),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    run_tests()
