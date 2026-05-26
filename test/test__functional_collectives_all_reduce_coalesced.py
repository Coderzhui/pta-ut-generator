# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._functional_collectives.all_reduce_coalesced 接口功能正确性
API 名称：torch.distributed._functional_collectives.all_reduce_coalesced
API 签名：def all_reduce_coalesced(self: list[torch.Tensor], reduceOp: str, group: RANK_TYPES, tag: str = "") -> list[torch.Tensor]

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 单元素列表 vs 多元素列表; tag="" vs 自定义 tag               | 已覆盖                                         |
| 枚举选项         | reduceOp: "sum", "avg", "product", "min", "max"             | 已覆盖                                         |
| 参数类型         | self=list[Tensor]; reduceOp=str; group=RANK_TYPES; tag=str   | 已覆盖                                         |
| 传参与不传参     | tag 默认空字符串 vs 显式传入                                 | 已覆盖                                         |
| 等价类/边界值    | 单张量列表 vs 多张量列表; 不同 dtype; 不同 shape             | 已覆盖                                         |
| 正常传参场景     | 各种 reduceOp、dtype、shape 组合                             | 已覆盖                                         |
| 异常传参场景     | 无稳定异常路径                                               | 未覆盖：API 要求合法分布式环境                 |
| 混合设备类型     | 所有张量在同一 NPU 设备上                                    | 未覆盖：不适用                                 |

未覆盖项及原因：
- 异常传参场景：all_reduce_coalesced 需要合法的分布式环境，参数校验依赖于后端
- 混合设备类型：all_reduce_coalesced 在同一进程组的同设备张量上操作

注意：本测试仅验证功能正确性（返回列表长度、张量 shape/dtype/device/type），
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
from torch.distributed._functional_collectives import all_reduce_coalesced


def _worker_fn(rank, world_size, device_name, result_queue):
    """Worker process entrypoint; runs all assertions and reports result."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29502'
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    try:
        # Collect assertion errors from each sub-test
        errors = []

        # --- Test 1: Single tensor in list, reduceOp="sum" ---
        try:
            tensors = [torch.ones(4, device=device_name)]
            result = all_reduce_coalesced(tensors, "sum", dist.group.WORLD)
            assert isinstance(result, list), f"Expected list, got {type(result)}"
            assert len(result) == 1, f"Expected list length 1, got {len(result)}"
            assert isinstance(result[0], torch.Tensor), "Expected Tensor element"
            assert result[0].shape == torch.Size([4]), f"Expected shape [4], got {result[0].shape}"
            assert result[0].dtype == torch.float32, f"Expected dtype float32, got {result[0].dtype}"
            assert str(result[0].device).startswith(device_name), f"Expected device {device_name}, got {result[0].device}"
        except Exception as e:
            errors.append(f"test_single_tensor_sum: {e}")

        # --- Test 2: Multiple tensors in list ---
        try:
            tensors = [
                torch.ones(3, device=device_name),
                torch.ones(2, 4, device=device_name),
            ]
            result = all_reduce_coalesced(tensors, "sum", dist.group.WORLD)
            assert isinstance(result, list), f"Expected list, got {type(result)}"
            assert len(result) == 2, f"Expected list length 2, got {len(result)}"
            # Verify each output tensor has same shape as corresponding input
            assert result[0].shape == torch.Size([3]), f"Expected shape [3], got {result[0].shape}"
            assert result[1].shape == torch.Size([2, 4]), f"Expected shape [2,4], got {result[1].shape}"
        except Exception as e:
            errors.append(f"test_multiple_tensors: {e}")

        # --- Test 3: reduceOp="avg" ---
        try:
            tensors = [torch.ones(4, device=device_name)]
            result = all_reduce_coalesced(tensors, "avg", dist.group.WORLD)
            assert isinstance(result, list), f"Expected list, got {type(result)}"
            assert len(result) == 1, f"Expected list length 1, got {len(result)}"
            assert result[0].shape == torch.Size([4]), f"Expected shape [4], got {result[0].shape}"
            assert result[0].dtype == torch.float32, f"Expected dtype float32, got {result[0].dtype}"
        except Exception as e:
            errors.append(f"test_reduceop_avg: {e}")

        # --- Test 4: reduceOp="product" ---
        try:
            tensors = [torch.ones(4, device=device_name)]
            result = all_reduce_coalesced(tensors, "product", dist.group.WORLD)
            assert isinstance(result, list), f"Expected list, got {type(result)}"
            assert len(result) == 1, f"Expected list length 1, got {len(result)}"
            assert result[0].shape == torch.Size([4]), f"Expected shape [4], got {result[0].shape}"
        except Exception as e:
            errors.append(f"test_reduceop_product: {e}")

        # --- Test 5: reduceOp="min" ---
        try:
            tensors = [torch.ones(4, device=device_name)]
            result = all_reduce_coalesced(tensors, "min", dist.group.WORLD)
            assert isinstance(result, list), f"Expected list, got {type(result)}"
            assert len(result) == 1, f"Expected list length 1, got {len(result)}"
            assert result[0].shape == torch.Size([4]), f"Expected shape [4], got {result[0].shape}"
        except Exception as e:
            errors.append(f"test_reduceop_min: {e}")

        # --- Test 6: reduceOp="max" ---
        try:
            tensors = [torch.ones(4, device=device_name)]
            result = all_reduce_coalesced(tensors, "max", dist.group.WORLD)
            assert isinstance(result, list), f"Expected list, got {type(result)}"
            assert len(result) == 1, f"Expected list length 1, got {len(result)}"
            assert result[0].shape == torch.Size([4]), f"Expected shape [4], got {result[0].shape}"
        except Exception as e:
            errors.append(f"test_reduceop_max: {e}")

        # --- Test 7: float16 dtype ---
        try:
            tensors = [torch.ones(4, dtype=torch.float16, device=device_name)]
            result = all_reduce_coalesced(tensors, "sum", dist.group.WORLD)
            assert isinstance(result, list), f"Expected list, got {type(result)}"
            assert len(result) == 1, f"Expected list length 1, got {len(result)}"
            assert result[0].dtype == torch.float16, f"Expected dtype float16, got {result[0].dtype}"
            assert result[0].shape == torch.Size([4]), f"Expected shape [4], got {result[0].shape}"
        except Exception as e:
            errors.append(f"test_dtype_float16: {e}")

        # --- Test 8: 2D shape ---
        try:
            tensors = [torch.ones(2, 3, device=device_name)]
            result = all_reduce_coalesced(tensors, "sum", dist.group.WORLD)
            assert isinstance(result, list), f"Expected list, got {type(result)}"
            assert len(result) == 1, f"Expected list length 1, got {len(result)}"
            assert result[0].shape == torch.Size([2, 3]), f"Expected shape [2,3], got {result[0].shape}"
        except Exception as e:
            errors.append(f"test_2d_shape: {e}")

        # --- Test 9: tag parameter with custom value ---
        try:
            tensors = [torch.ones(4, device=device_name)]
            result = all_reduce_coalesced(tensors, "sum", dist.group.WORLD, tag="test_tag")
            assert isinstance(result, list), f"Expected list, got {type(result)}"
            assert len(result) == 1, f"Expected list length 1, got {len(result)}"
            assert result[0].shape == torch.Size([4]), f"Expected shape [4], got {result[0].shape}"
        except Exception as e:
            errors.append(f"test_custom_tag: {e}")

        # --- Test 10: empty tag (default) ---
        try:
            tensors = [torch.ones(4, device=device_name)]
            result = all_reduce_coalesced(tensors, "sum", dist.group.WORLD, tag="")
            assert isinstance(result, list), f"Expected list, got {type(result)}"
            assert len(result) == 1, f"Expected list length 1, got {len(result)}"
            assert result[0].shape == torch.Size([4]), f"Expected shape [4], got {result[0].shape}"
        except Exception as e:
            errors.append(f"test_empty_tag: {e}")

        # --- Test 11: return type is list of tensors with matching shapes ---
        try:
            tensors = [
                torch.ones(2, device=device_name),
                torch.ones(3, 4, device=device_name),
                torch.ones(5, 6, 7, device=device_name),
            ]
            result = all_reduce_coalesced(tensors, "sum", dist.group.WORLD)
            assert isinstance(result, list), f"Expected list, got {type(result)}"
            assert len(result) == 3, f"Expected list length 3, got {len(result)}"
            assert result[0].shape == torch.Size([2]), f"Shape mismatch for tensor 0"
            assert result[1].shape == torch.Size([3, 4]), f"Shape mismatch for tensor 1"
            assert result[2].shape == torch.Size([5, 6, 7]), f"Shape mismatch for tensor 2"
            # Verify all elements are tensors
            for i, t in enumerate(result):
                assert isinstance(t, torch.Tensor), f"Element {i} is not a Tensor"
                assert str(t.device).startswith(device_name), f"Element {i} device mismatch"
        except Exception as e:
            errors.append(f"test_return_type_shapes: {e}")

        # Report results from rank 0
        if rank == 0:
            if errors:
                result_queue.put(("FAIL", errors))
            else:
                result_queue.put(("PASS", None))
    finally:
        dist.destroy_process_group()


class TestFunctionalCollectivesAllReduceCoalesced(TestCase):
    """Test cases for torch.distributed._functional_collectives.all_reduce_coalesced."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_all_reduce_coalesced(self):
        """Comprehensive test for all_reduce_coalesced covering all dimensions."""
        world_size = 2
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        mp.spawn(
            _worker_fn,
            args=(world_size, self.device_name, result_queue),
            nprocs=world_size,
            join=True
        )
        status, detail = result_queue.get(timeout=30)
        if status == "FAIL":
            self.fail(f"Worker assertions failed: {detail}")


if __name__ == "__main__":
    run_tests()
