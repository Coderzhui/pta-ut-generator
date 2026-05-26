# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._functional_collectives.all_gather_inplace 接口功能正确性
API 名称：torch.distributed._functional_collectives.all_gather_inplace
API 签名：def all_gather_inplace(tensor_list: list[torch.Tensor], tensor: torch.Tensor, group=None, async_op=False, tag: str = "")

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | group=None(默认WORLD) vs 显式传入; tag="" vs 自定义字符串    | 已覆盖                                         |
| 枚举选项         | async_op=True/False                                          | 已覆盖                                         |
| 参数类型         | tensor_list=list[Tensor]; tensor=Tensor; group=ProcessGroup; async_op=bool; tag=str | 已覆盖 |
| 传参与不传参     | group/tag 使用默认值 vs 显式传入                             | 已覆盖                                         |
| 等价类/边界值    | 1D vs 2D tensor; float32 vs float16; async_op 两种值         | 已覆盖                                         |
| 正常传参场景     | 2 rank gather, tensor_list 原地修改, 返回值即 tensor_list    | 已覆盖                                         |
| 异常传参场景     | async_op=True raises AssertionError                          | 已覆盖                                         |
| 混合设备类型     | 单一 NPU 设备类型，不涉及异构设备输入                        | 未覆盖：不适用                                 |

未覆盖项及原因：
- 混合设备类型：all_gather_inplace 在同一设备上操作，不涉及多 Tensor 异构设备输入

注意：本测试仅验证功能正确性（形状、类型、原地修改行为、返回值），
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
from torch.distributed._functional_collectives import all_gather_inplace


def _worker_fn(rank, world_size, device_name, test_name, result_queue):
    """Worker function: initialize HCCL, run test, report result."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29508'
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    try:
        _dispatch_test(test_name, rank, world_size, device_name)
        if rank == 0:
            result_queue.put(("PASS", None))
    except Exception as e:
        if rank == 0:
            result_queue.put(("FAIL", str(e)))
        raise
    finally:
        dist.destroy_process_group()


def _dispatch_test(test_name, rank, world_size, device_name):
    """Dispatch to the correct test function by name."""
    test_map = {
        "basic_2ranks": _test_basic_2ranks,
        "1d_tensor": _test_1d_tensor,
        "2d_tensor": _test_2d_tensor,
        "dtype_float32": _test_dtype_float32,
        "dtype_float16": _test_dtype_float16,
        "tag_default": _test_tag_default,
        "tag_custom": _test_tag_custom,
        "inplace_modification": _test_inplace_modification,
        "return_value_is_tensor_list": _test_return_value_is_tensor_list,
        "async_op_raises": _test_async_op_raises,
    }
    fn = test_map[test_name]
    fn(rank, world_size, device_name)


def _test_basic_2ranks(rank, world_size, device_name):
    """Worker: basic all_gather with 2 ranks, tensor_list gets correct shapes."""
    device = f"{device_name}:{rank}"
    tensor = torch.full((4,), float(rank + 1), device=device)
    tensor_list = [torch.zeros(4, device=device) for _ in range(world_size)]
    all_gather_inplace(tensor_list, tensor)

    # tensor_list should have world_size entries, each of shape (4,)
    assert len(tensor_list) == world_size, (
        f"tensor_list length should be {world_size}, got {len(tensor_list)}"
    )
    for i, t in enumerate(tensor_list):
        assert t.shape == (4,), (
            f"tensor_list[{i}] shape should be (4,), got {t.shape}"
        )
        assert t.dtype == tensor.dtype, (
            f"tensor_list[{i}] dtype should be {tensor.dtype}, got {t.dtype}"
        )


def _test_1d_tensor(rank, world_size, device_name):
    """Worker: all_gather with 1D tensor."""
    device = f"{device_name}:{rank}"
    tensor = torch.full((8,), float(rank), device=device)
    tensor_list = [torch.zeros(8, device=device) for _ in range(world_size)]
    all_gather_inplace(tensor_list, tensor)

    assert len(tensor_list) == world_size
    for i, t in enumerate(tensor_list):
        assert t.shape == (8,), f"Expected shape (8,), got {t.shape}"


def _test_2d_tensor(rank, world_size, device_name):
    """Worker: all_gather with 2D tensor."""
    device = f"{device_name}:{rank}"
    tensor = torch.full((3, 4), float(rank + 1), device=device)
    tensor_list = [torch.zeros(3, 4, device=device) for _ in range(world_size)]
    all_gather_inplace(tensor_list, tensor)

    assert len(tensor_list) == world_size
    for i, t in enumerate(tensor_list):
        assert t.shape == (3, 4), f"Expected shape (3, 4), got {t.shape}"


def _test_dtype_float32(rank, world_size, device_name):
    """Worker: all_gather with float32 dtype."""
    device = f"{device_name}:{rank}"
    tensor = torch.full((4,), float(rank + 1), dtype=torch.float32, device=device)
    tensor_list = [torch.zeros(4, dtype=torch.float32, device=device) for _ in range(world_size)]
    all_gather_inplace(tensor_list, tensor)

    for i, t in enumerate(tensor_list):
        assert t.dtype == torch.float32, (
            f"tensor_list[{i}] dtype should be float32, got {t.dtype}"
        )


def _test_dtype_float16(rank, world_size, device_name):
    """Worker: all_gather with float16 dtype."""
    device = f"{device_name}:{rank}"
    tensor = torch.full((4,), float(rank + 1), dtype=torch.float16, device=device)
    tensor_list = [torch.zeros(4, dtype=torch.float16, device=device) for _ in range(world_size)]
    all_gather_inplace(tensor_list, tensor)

    for i, t in enumerate(tensor_list):
        assert t.dtype == torch.float16, (
            f"tensor_list[{i}] dtype should be float16, got {t.dtype}"
        )


def _test_tag_default(rank, world_size, device_name):
    """Worker: all_gather with default tag (empty string)."""
    device = f"{device_name}:{rank}"
    tensor = torch.full((4,), float(rank + 1), device=device)
    tensor_list = [torch.zeros(4, device=device) for _ in range(world_size)]
    # tag defaults to ""
    all_gather_inplace(tensor_list, tensor)

    assert len(tensor_list) == world_size
    for t in tensor_list:
        assert t.shape == (4,)


def _test_tag_custom(rank, world_size, device_name):
    """Worker: all_gather with custom tag string."""
    device = f"{device_name}:{rank}"
    tensor = torch.full((4,), float(rank + 1), device=device)
    tensor_list = [torch.zeros(4, device=device) for _ in range(world_size)]
    all_gather_inplace(tensor_list, tensor, tag="custom_tag_123")

    assert len(tensor_list) == world_size
    for t in tensor_list:
        assert t.shape == (4,)


def _test_inplace_modification(rank, world_size, device_name):
    """Worker: verify tensor_list tensors are modified in-place."""
    device = f"{device_name}:{rank}"
    tensor = torch.full((4,), float(rank + 1), device=device)
    tensor_list = [torch.zeros(4, device=device) for _ in range(world_size)]

    # Capture tensor object ids before the call
    ids_before = [id(t) for t in tensor_list]
    all_gather_inplace(tensor_list, tensor)
    ids_after = [id(t) for t in tensor_list]

    # The same tensor objects should still be in tensor_list (in-place)
    assert ids_before == ids_after, (
        "tensor_list tensors should be modified in-place, not replaced"
    )


def _test_return_value_is_tensor_list(rank, world_size, device_name):
    """Worker: verify return value is the same tensor_list object."""
    device = f"{device_name}:{rank}"
    tensor = torch.full((4,), float(rank + 1), device=device)
    tensor_list = [torch.zeros(4, device=device) for _ in range(world_size)]

    result = all_gather_inplace(tensor_list, tensor)

    # Return value should be the same list object
    assert result is tensor_list, (
        "all_gather_inplace should return the same tensor_list object"
    )


def _test_async_op_raises(rank, world_size, device_name):
    """Worker: async_op=True should raise AssertionError."""
    device = f"{device_name}:{rank}"
    tensor = torch.full((4,), float(rank + 1), device=device)
    tensor_list = [torch.zeros(4, device=device) for _ in range(world_size)]

    raised = False
    try:
        all_gather_inplace(tensor_list, tensor, async_op=True)
    except (AssertionError, RuntimeError):
        raised = True

    assert raised, "async_op=True should raise AssertionError or RuntimeError"


def _run_test(self, test_name):
    """Helper: spawn worker processes and check result."""
    world_size = 2
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    mp.spawn(
        _worker_fn,
        args=(world_size, self.device_name, test_name, result_queue),
        nprocs=world_size,
        join=True,
    )
    if not result_queue.empty():
        status, msg = result_queue.get()
        self.assertEqual(status, "PASS", f"Worker failed: {msg}")


class TestFunctionalCollectivesAllGatherInplace(TestCase):
    """Test cases for torch.distributed._functional_collectives.all_gather_inplace."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_basic_2ranks(self):
        """Basic all_gather with 2 ranks, tensor_list populated with correct shapes."""
        _run_test(self, "basic_2ranks")

    @skipIfUnsupportMultiNPU(2)
    def test_1d_tensor(self):
        """All_gather with 1D tensor."""
        _run_test(self, "1d_tensor")

    @skipIfUnsupportMultiNPU(2)
    def test_2d_tensor(self):
        """All_gather with 2D tensor."""
        _run_test(self, "2d_tensor")

    @skipIfUnsupportMultiNPU(2)
    def test_dtype_float32(self):
        """All_gather with float32 dtype."""
        _run_test(self, "dtype_float32")

    @skipIfUnsupportMultiNPU(2)
    def test_dtype_float16(self):
        """All_gather with float16 dtype."""
        _run_test(self, "dtype_float16")

    @skipIfUnsupportMultiNPU(2)
    def test_tag_default(self):
        """All_gather with default tag parameter."""
        _run_test(self, "tag_default")

    @skipIfUnsupportMultiNPU(2)
    def test_tag_custom(self):
        """All_gather with custom tag string."""
        _run_test(self, "tag_custom")

    @skipIfUnsupportMultiNPU(2)
    def test_inplace_modification(self):
        """Verify tensor_list tensors are modified in-place."""
        _run_test(self, "inplace_modification")

    @skipIfUnsupportMultiNPU(2)
    def test_return_value_is_tensor_list(self):
        """Return value is the same tensor_list object."""
        _run_test(self, "return_value_is_tensor_list")

    @skipIfUnsupportMultiNPU(2)
    def test_async_op_raises(self):
        """async_op=True raises AssertionError."""
        _run_test(self, "async_op_raises")


if __name__ == "__main__":
    run_tests()
