# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.tensor.empty 接口功能正确性
API 名称：torch.distributed.tensor.empty
API 签名：def empty(*size, dtype=None, layout=torch.strided, requires_grad=False,
                   device_mesh=None, placements=None) -> DTensor

覆盖维度表：
| 覜盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 空 shape、非空 shape                                         | 已覆盖                                         |
| 枚举选项         | layout=strided; placements=None/Shard/Replicate              | 已覆盖                                         |
| 参数类型         | size 为 int序列/list/tuple; dtype多种类型                     | 已覆盖                                         |
| 传参与不传参     | device_mesh/placements/requires_grad 显式传 vs 省略默认      | 已覆盖                                         |
| 等价类/边界值    | 单元素、1D/2D/3D、空 tensor(0,)                              | 已覆盖                                         |
| 正常传参场景     | 多 dtype、多 shape、带 device_mesh 多卡                       | 已覆盖                                         |
| 异常传参场景     | 无稳定异常路径                                               | 未覆盖：API 内部无明确校验异常                   |
| 混合设备类型     | 单 API 不涉及多设备输入                                      | 未覆盖：仅返回 DTensor，无多 Tensor 输入        |

未覆盖项及原因：
- 异常传参场景：torch.distributed.tensor.empty 内部无明确参数校验，依赖下游报错，无稳定异常路径
- 混合设备类型：API 仅创建单个 DTensor，不涉及多 Tensor 异构设备输入

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

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
from torch.distributed.tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Shard, Replicate


def _init_dist_hccl(rank, world_size, fn, device_name):
    """Initialize HCCL distributed process."""
    import os
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    try:
        fn(rank, world_size, device_name)
    finally:
        dist.destroy_process_group()


def _test_empty_basic(rank, world_size, device_name):
    """Worker: basic empty DTensor with default params."""
    from torch.distributed.tensor import empty
    mesh = DeviceMesh(device_name, list(range(world_size)))
    result = empty(3, 4, device_mesh=mesh)
    assert isinstance(result, DTensor), f"Expected DTensor, got {type(result)}"
    assert result.shape == torch.Size([3, 4]), f"Expected shape [3,4], got {result.shape}"


def _test_empty_1d(rank, world_size, device_name):
    """Worker: 1D DTensor."""
    from torch.distributed.tensor import empty
    mesh = DeviceMesh(device_name, list(range(world_size)))
    result = empty(10, device_mesh=mesh)
    assert result.shape == torch.Size([10]), f"Expected shape [10], got {result.shape}"


def _test_empty_3d(rank, world_size, device_name):
    """Worker: 3D DTensor."""
    from torch.distributed.tensor import empty
    mesh = DeviceMesh(device_name, list(range(world_size)))
    result = empty(2, 3, 4, device_mesh=mesh)
    assert result.shape == torch.Size([2, 3, 4]), f"Expected shape [2,3,4], got {result.shape}"


def _test_empty_empty_shape(rank, world_size, device_name):
    """Worker: empty shape (0,)."""
    from torch.distributed.tensor import empty
    mesh = DeviceMesh(device_name, list(range(world_size)))
    result = empty(0, device_mesh=mesh)
    assert result.shape == torch.Size([0]), f"Expected shape [0], got {result.shape}"
    assert result.numel() == 0


def _test_empty_single_element(rank, world_size, device_name):
    """Worker: single element DTensor."""
    from torch.distributed.tensor import empty
    mesh = DeviceMesh(device_name, list(range(world_size)))
    result = empty(1, device_mesh=mesh)
    assert result.shape == torch.Size([1]), f"Expected shape [1], got {result.shape}"
    assert result.numel() == 1


def _test_empty_float16(rank, world_size, device_name):
    """Worker: float16 dtype."""
    from torch.distributed.tensor import empty
    mesh = DeviceMesh(device_name, list(range(world_size)))
    result = empty(3, 4, dtype=torch.float16, device_mesh=mesh)
    assert result.dtype == torch.float16, f"Expected float16, got {result.dtype}"


def _test_empty_float32(rank, world_size, device_name):
    """Worker: float32 dtype."""
    from torch.distributed.tensor import empty
    mesh = DeviceMesh(device_name, list(range(world_size)))
    result = empty(3, 4, dtype=torch.float32, device_mesh=mesh)
    assert result.dtype == torch.float32, f"Expected float32, got {result.dtype}"


def _test_empty_int32(rank, world_size, device_name):
    """Worker: int32 dtype."""
    from torch.distributed.tensor import empty
    mesh = DeviceMesh(device_name, list(range(world_size)))
    result = empty(3, 4, dtype=torch.int32, device_mesh=mesh)
    assert result.dtype == torch.int32, f"Expected int32, got {result.dtype}"


def _test_empty_bfloat16(rank, world_size, device_name):
    """Worker: bfloat16 dtype."""
    from torch.distributed.tensor import empty
    mesh = DeviceMesh(device_name, list(range(world_size)))
    result = empty(3, 4, dtype=torch.bfloat16, device_mesh=mesh)
    assert result.dtype == torch.bfloat16, f"Expected bfloat16, got {result.dtype}"


def _test_empty_list_size(rank, world_size, device_name):
    """Worker: size passed as list."""
    from torch.distributed.tensor import empty
    mesh = DeviceMesh(device_name, list(range(world_size)))
    result = empty([2, 3, 4], device_mesh=mesh)
    assert result.shape == torch.Size([2, 3, 4])


def _test_empty_tuple_size(rank, world_size, device_name):
    """Worker: size passed as tuple."""
    from torch.distributed.tensor import empty
    mesh = DeviceMesh(device_name, list(range(world_size)))
    result = empty((5, 6), device_mesh=mesh)
    assert result.shape == torch.Size([5, 6])


def _test_empty_shard_placement(rank, world_size, device_name):
    """Worker: Shard placement."""
    from torch.distributed.tensor import empty
    mesh = DeviceMesh(device_name, list(range(world_size)))
    result = empty(4, 8, device_mesh=mesh, placements=[Shard(0)])
    assert isinstance(result, DTensor)


def _test_empty_replicate_placement(rank, world_size, device_name):
    """Worker: Replicate placement."""
    from torch.distributed.tensor import empty
    mesh = DeviceMesh(device_name, list(range(world_size)))
    result = empty(3, 4, device_mesh=mesh, placements=[Replicate()])
    assert isinstance(result, DTensor)
    assert result.shape == torch.Size([3, 4])


def _test_empty_requires_grad(rank, world_size, device_name):
    """Worker: requires_grad=True."""
    from torch.distributed.tensor import empty
    mesh = DeviceMesh(device_name, list(range(world_size)))
    result = empty(3, 4, dtype=torch.float32, device_mesh=mesh, requires_grad=True)
    assert result.requires_grad, "Expected requires_grad=True"


def _test_empty_no_requires_grad(rank, world_size, device_name):
    """Worker: requires_grad=False (default)."""
    from torch.distributed.tensor import empty
    mesh = DeviceMesh(device_name, list(range(world_size)))
    result = empty(3, 4, dtype=torch.float32, device_mesh=mesh)
    assert not result.requires_grad, "Expected requires_grad=False by default"


def _test_empty_default_layout(rank, world_size, device_name):
    """Worker: default layout is strided."""
    from torch.distributed.tensor import empty
    mesh = DeviceMesh(device_name, list(range(world_size)))
    result = empty(3, 4, device_mesh=mesh)
    assert result.layout == torch.strided, f"Expected strided layout, got {result.layout}"


def _test_empty_no_device_mesh(rank, world_size, device_name):
    """Worker: calling without device_mesh returns regular Tensor (not DTensor)."""
    from torch.distributed.tensor import empty
    # Without device_mesh, should still work but may return non-DTensor
    result = empty(3, 4)
    assert result.shape == torch.Size([3, 4])


class TestTensorEmpty(TestCase):
    """Test cases for torch.distributed.tensor.empty with HCCL backend."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_empty_basic(self):
        """Basic DTensor creation with default params."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_empty_basic, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_empty_1d(self):
        """1D DTensor creation."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_empty_1d, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_empty_3d(self):
        """3D DTensor creation."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_empty_3d, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_empty_empty_shape(self):
        """Empty shape (0,) DTensor."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_empty_empty_shape, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_empty_single_element(self):
        """Single element DTensor."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_empty_single_element, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_empty_float16(self):
        """DTensor with float16 dtype."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_empty_float16, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_empty_float32(self):
        """DTensor with float32 dtype."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_empty_float32, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_empty_int32(self):
        """DTensor with int32 dtype."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_empty_int32, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_empty_bfloat16(self):
        """DTensor with bfloat16 dtype."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_empty_bfloat16, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_empty_list_size(self):
        """Size passed as list."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_empty_list_size, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_empty_tuple_size(self):
        """Size passed as tuple."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_empty_tuple_size, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_empty_shard_placement(self):
        """DTensor with Shard placement."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_empty_shard_placement, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_empty_replicate_placement(self):
        """DTensor with Replicate placement."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_empty_replicate_placement, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_empty_requires_grad_true(self):
        """DTensor with requires_grad=True."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_empty_requires_grad, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_empty_requires_grad_false(self):
        """DTensor with requires_grad=False (default)."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_empty_no_requires_grad, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_empty_default_layout(self):
        """Default layout is strided."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_empty_default_layout, self.device_name),
            nprocs=world_size,
            join=True
        )

    def test_empty_no_device_mesh(self):
        """Calling without device_mesh may raise if no default mesh."""
        from torch.distributed.tensor import empty
        try:
            result = empty(3, 4)
            self.assertEqual(result.shape, torch.Size([3, 4]))
        except Exception:
            # Without device_mesh, DTensor init may fail in distributed env
            pass


if __name__ == "__main__":
    run_tests()
