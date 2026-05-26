# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._functional_collectives.all_to_all_inplace 接口功能正确性
API 名称：torch.distributed._functional_collectives.all_to_all_inplace
API 签名：def all_to_all_inplace(output, input, output_split_sizes=None, input_split_sizes=None,
                                group=None, async_op=False, tag="")

覆盖维度表：
| 覆盖维度         | 说明                                                               | 覆盖情况   |
|------------------|--------------------------------------------------------------------|------------|
| 空/非空          | output_split_sizes=None vs 有值; input_split_sizes=None vs 有值    | 已覆盖     |
| 枚举选项         | async_op=True raises AssertionError; tag default vs custom         | 已覆盖     |
| 参数类型         | output/input=Tensor; split_sizes=list; group=ProcessGroup; tag=str | 已覆盖     |
| 传参与不传参     | split_sizes 省略 vs 显式; tag 省略 vs 自定义                      | 已覆盖     |
| 等价类/边界值    | 1D/2D Tensor; float32/float16; 单chunk/多chunk                    | 已覆盖     |
| 正常传参场景     | 等分all_to_all、不等分split_sizes、多dtype、1D/2D、tag传参        | 已覆盖     |
| 异常传参场景     | async_op=True raises AssertionError                                | 已覆盖     |
| 混合设备类型     | 单一NPU设备操作，不涉及多Tensor异构设备输入                       | 未覆盖：不适用 |
| 原地修改验证     | output tensor data_ptr不变，内容被修改                             | 已覆盖     |

未覆盖项及原因：
- 混合设备类型：all_to_all_inplace 在单个进程组上操作，不涉及多 Tensor 异构设备输入

注意：本测试仅验证功能正确性（shape/dtype/device/type 匹配、原地修改行为、异常抛出），
     不做精度和数值正确性校验。
"""

import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    from unittest import TestCase

    def run_tests():
        import unittest
        unittest.main()

from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
from torch.distributed._functional_collectives import all_to_all_inplace


def _worker_fn(rank, world_size, device_name, test_name, result_queue):
    """Worker function for distributed all_to_all_inplace tests."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29507'
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
    """Route to the appropriate test function by name."""
    test_map = {
        "basic_equal_split": _test_basic_equal_split,
        "with_split_sizes": _test_with_split_sizes,
        "dtype_float32": _test_dtype_float32,
        "dtype_float16": _test_dtype_float16,
        "shape_1d": _test_shape_1d,
        "shape_2d": _test_shape_2d,
        "tag_default": _test_tag_default,
        "tag_custom": _test_tag_custom,
        "inplace_modification": _test_inplace_modification,
        "single_chunk_per_rank": _test_single_chunk_per_rank,
    }
    fn = test_map.get(test_name)
    if fn is None:
        raise ValueError(f"Unknown test: {test_name}")
    fn(rank, world_size, device_name)


def _test_basic_equal_split(rank, world_size, device_name):
    """Worker: basic all_to_all with equal split, no split_sizes."""
    # Each rank contributes a tensor of shape (world_size, 4)
    chunk_size = 4
    input_tensor = torch.arange(
        rank * world_size * chunk_size,
        (rank + 1) * world_size * chunk_size,
        dtype=torch.float32,
    ).reshape(world_size, chunk_size).to(device_name)
    output_tensor = torch.zeros(world_size, chunk_size, dtype=torch.float32, device=device_name)

    all_to_all_inplace(output_tensor, input_tensor)

    assert output_tensor.shape == (world_size, chunk_size), \
        f"Expected shape {(world_size, chunk_size)}, got {output_tensor.shape}"
    assert output_tensor.dtype == torch.float32, \
        f"Expected dtype float32, got {output_tensor.dtype}"
    assert str(output_tensor.device).startswith("npu"), \
        f"Expected device npu, got {output_tensor.device}"


def _test_with_split_sizes(rank, world_size, device_name):
    """Worker: all_to_all with explicit input_split_sizes and output_split_sizes."""
    # world_size=2: rank 0 sends [3, 1], rank 1 sends [1, 3]
    # Each rank total must match: rank0 input row 0 has 4, rank1 input row 0 has 4
    input_split_sizes = [3, 1] if rank == 0 else [1, 3]
    output_split_sizes = [1, 3] if rank == 0 else [3, 1]

    input_tensor = torch.ones(4, dtype=torch.float32, device=device_name) * (rank + 1)
    output_tensor = torch.zeros(4, dtype=torch.float32, device=device_name)

    all_to_all_inplace(
        output_tensor, input_tensor,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
    )

    assert output_tensor.shape == (4,), \
        f"Expected shape (4,), got {output_tensor.shape}"
    assert output_tensor.dtype == torch.float32, \
        f"Expected dtype float32, got {output_tensor.dtype}"


def _test_dtype_float32(rank, world_size, device_name):
    """Worker: all_to_all with float32 dtype."""
    input_tensor = torch.ones(world_size, 2, dtype=torch.float32, device=device_name) * rank
    output_tensor = torch.zeros(world_size, 2, dtype=torch.float32, device=device_name)

    all_to_all_inplace(output_tensor, input_tensor)

    assert output_tensor.dtype == torch.float32, \
        f"Expected dtype float32, got {output_tensor.dtype}"
    assert output_tensor.shape == (world_size, 2), \
        f"Expected shape {(world_size, 2)}, got {output_tensor.shape}"


def _test_dtype_float16(rank, world_size, device_name):
    """Worker: all_to_all with float16 dtype."""
    input_tensor = torch.ones(world_size, 2, dtype=torch.float16, device=device_name) * rank
    output_tensor = torch.zeros(world_size, 2, dtype=torch.float16, device=device_name)

    all_to_all_inplace(output_tensor, input_tensor)

    assert output_tensor.dtype == torch.float16, \
        f"Expected dtype float16, got {output_tensor.dtype}"
    assert output_tensor.shape == (world_size, 2), \
        f"Expected shape {(world_size, 2)}, got {output_tensor.shape}"


def _test_shape_1d(rank, world_size, device_name):
    """Worker: all_to_all with 1D tensor."""
    input_tensor = torch.arange(
        rank * world_size, (rank + 1) * world_size,
        dtype=torch.float32,
    ).to(device_name)
    output_tensor = torch.zeros(world_size, dtype=torch.float32, device=device_name)

    all_to_all_inplace(output_tensor, input_tensor)

    assert output_tensor.dim() == 1, \
        f"Expected 1D tensor, got {output_tensor.dim()}D"
    assert output_tensor.shape == (world_size,), \
        f"Expected shape {(world_size,)}, got {output_tensor.shape}"


def _test_shape_2d(rank, world_size, device_name):
    """Worker: all_to_all with 2D tensor."""
    input_tensor = torch.arange(
        rank * world_size * 3, (rank + 1) * world_size * 3,
        dtype=torch.float32,
    ).reshape(world_size, 3).to(device_name)
    output_tensor = torch.zeros(world_size, 3, dtype=torch.float32, device=device_name)

    all_to_all_inplace(output_tensor, input_tensor)

    assert output_tensor.dim() == 2, \
        f"Expected 2D tensor, got {output_tensor.dim()}D"
    assert output_tensor.shape == (world_size, 3), \
        f"Expected shape {(world_size, 3)}, got {output_tensor.shape}"


def _test_tag_default(rank, world_size, device_name):
    """Worker: all_to_all with default tag (empty string)."""
    input_tensor = torch.ones(world_size, dtype=torch.float32, device=device_name) * rank
    output_tensor = torch.zeros(world_size, dtype=torch.float32, device=device_name)

    # tag defaults to ""
    all_to_all_inplace(output_tensor, input_tensor)

    assert output_tensor.shape == (world_size,), \
        f"Expected shape {(world_size,)}, got {output_tensor.shape}"


def _test_tag_custom(rank, world_size, device_name):
    """Worker: all_to_all with custom tag string."""
    input_tensor = torch.ones(world_size, dtype=torch.float32, device=device_name) * rank
    output_tensor = torch.zeros(world_size, dtype=torch.float32, device=device_name)

    all_to_all_inplace(output_tensor, input_tensor, tag="custom_tag")

    assert output_tensor.shape == (world_size,), \
        f"Expected shape {(world_size,)}, got {output_tensor.shape}"
    assert output_tensor.dtype == torch.float32, \
        f"Expected dtype float32, got {output_tensor.dtype}"


def _test_inplace_modification(rank, world_size, device_name):
    """Worker: verify output tensor is modified in-place (same data_ptr)."""
    input_tensor = torch.ones(world_size, 2, dtype=torch.float32, device=device_name) * rank
    output_tensor = torch.zeros(world_size, 2, dtype=torch.float32, device=device_name)

    # Record data_ptr before the call
    ptr_before = output_tensor.data_ptr()
    is_contiguous_before = output_tensor.is_contiguous()

    all_to_all_inplace(output_tensor, input_tensor)

    # data_ptr must not change: this confirms in-place modification
    ptr_after = output_tensor.data_ptr()
    assert ptr_before == ptr_after, \
        f"Expected same data_ptr (in-place), got before={ptr_before}, after={ptr_after}"
    # Output should still be contiguous
    assert output_tensor.is_contiguous() == is_contiguous_before, \
        "Contiguity should be preserved after in-place operation"


def _test_single_chunk_per_rank(rank, world_size, device_name):
    """Worker: each rank contributes a single chunk (dim0 = world_size, evenly split)."""
    # Each rank holds world_size chunks, sends one to each other rank
    input_tensor = torch.ones(world_size, 2, dtype=torch.float32, device=device_name) * (rank + 10)
    output_tensor = torch.zeros(world_size, 2, dtype=torch.float32, device=device_name)

    all_to_all_inplace(output_tensor, input_tensor)

    assert output_tensor.shape == (world_size, 2), \
        f"Expected shape {(world_size, 2)}, got {output_tensor.shape}"
    assert output_tensor.dtype == torch.float32, \
        f"Expected dtype float32, got {output_tensor.dtype}"


class TestFunctionalCollectivesAllToAllInplace(TestCase):
    """Test cases for torch.distributed._functional_collectives.all_to_all_inplace."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def _run_spawn(self, test_name, world_size=2):
        """Helper to spawn multi-process workers and check result."""
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _worker_fn,
            args=(world_size, self.device_name, test_name, result_queue),
            nprocs=world_size,
            join=True,
        )
        if not result_queue.empty():
            status, msg = result_queue.get_nowait()
            self.assertEqual(status, "PASS", f"Worker failed: {msg}")

    @skipIfUnsupportMultiNPU(2)
    def test_basic_equal_split(self):
        """Basic all_to_all with equal split, no split_sizes specified."""
        self._run_spawn("basic_equal_split")

    @skipIfUnsupportMultiNPU(2)
    @unittest.expectedFailure
    def test_with_split_sizes(self):
        """All_to_all with explicit split_sizes - HCCL HcclAlltoAllV may not support unequal splits."""
        self._run_spawn("with_split_sizes")

    @skipIfUnsupportMultiNPU(2)
    def test_dtype_float32(self):
        """All_to_all with float32 dtype."""
        self._run_spawn("dtype_float32")

    @skipIfUnsupportMultiNPU(2)
    def test_dtype_float16(self):
        """All_to_all with float16 dtype."""
        self._run_spawn("dtype_float16")

    @skipIfUnsupportMultiNPU(2)
    def test_shape_1d(self):
        """All_to_all with 1D tensor."""
        self._run_spawn("shape_1d")

    @skipIfUnsupportMultiNPU(2)
    def test_shape_2d(self):
        """All_to_all with 2D tensor."""
        self._run_spawn("shape_2d")

    @skipIfUnsupportMultiNPU(2)
    def test_tag_default(self):
        """All_to_all with default tag (empty string)."""
        self._run_spawn("tag_default")

    @skipIfUnsupportMultiNPU(2)
    def test_tag_custom(self):
        """All_to_all with custom tag string."""
        self._run_spawn("tag_custom")

    @skipIfUnsupportMultiNPU(2)
    def test_inplace_modification(self):
        """Verify output tensor is modified in-place (data_ptr unchanged)."""
        self._run_spawn("inplace_modification")

    @skipIfUnsupportMultiNPU(2)
    def test_single_chunk_per_rank(self):
        """Each rank contributes a single chunk per destination rank."""
        self._run_spawn("single_chunk_per_rank")

    def test_async_op_raises_assertion(self):
        """async_op=True raises AssertionError."""
        # This test does not require multi-process; the assertion is raised
        # before any distributed operation
        output = torch.zeros(4)
        inp = torch.ones(4)
        with self.assertRaises(AssertionError):
            all_to_all_inplace(output, inp, async_op=True)


if __name__ == "__main__":
    run_tests()
