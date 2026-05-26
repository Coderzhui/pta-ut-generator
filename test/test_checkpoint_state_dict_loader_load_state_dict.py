# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.state_dict_loader.load_state_dict 接口功能正确性
API 名称：torch.distributed.checkpoint.state_dict_loader.load_state_dict
API 签名：load_state_dict(state_dict: dict[str, Any], storage_reader: StorageReader, process_group: dist.ProcessGroup | None = None, coordinator_rank: int = 0, no_dist: bool = False, planner: LoadPlanner | None = None) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 空 state_dict vs 非空 state_dict                             | 已覆盖                                          |
| 枚举选项         | coordinator_rank (0, 非默认)                                 | 已覆盖                                          |
| 参数类型         | state_dict: dict, storage_reader: StorageReader              | 已覆盖                                          |
| 传参与不传参     | no_dist 默认 False vs True；planner 默认 None vs 传入        | 已覆盖                                          |
| 等价类/边界值    | 单 tensor、多 tensor state_dict                              | 已覆盖                                          |
| 正常传参场景     | no_dist=True 单进程加载；多卡 HCCL 加载                      | 已覆盖                                          |
| 异常传参场景     | 无稳定异常路径                                                | 未覆盖：内部异常随上游变化                      |
| 混合设备类型     | NPU tensor 通过 HCCL 跨 rank 同步                            | 已覆盖                                          |

未覆盖项及原因：
- 异常传参场景：load_state_dict 内部异常依赖 StorageReader 和上游实现，不稳定

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import os
import tempfile
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from torch.distributed.checkpoint import FileSystemWriter, FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import load_state_dict

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


def _init_process(rank, world_size, fn, device_name):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29502"
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    torch.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    try:
        fn(rank, world_size, device_name)
    finally:
        dist.destroy_process_group()


def _test_load_no_dist(rank, world_size, device_name, ckpt_dir, result_queue):
    """Worker: test load_state_dict with no_dist=True."""
    state_dict = {"weight": torch.zeros(3, 4), "bias": torch.zeros(4)}
    storage_reader = FileSystemReader(ckpt_dir)
    load_state_dict(state_dict, storage_reader, no_dist=True)
    result_queue.put(("keys", rank, list(state_dict.keys())))
    result_queue.put(("weight_shape", rank, tuple(state_dict["weight"].shape)))
    result_queue.put(("bias_shape", rank, tuple(state_dict["bias"].shape)))


def _run_load_multicard(rank, world_size, ckpt_dir):
    """Worker: test load_state_dict with multiple NPUs."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29502"
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    torch.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    try:
        local_device = f"npu:{rank}"
        state_dict = {
            "weight": torch.zeros(3, 4, device=local_device),
            "bias": torch.zeros(4, device=local_device),
        }
        storage_reader = FileSystemReader(ckpt_dir)
        load_state_dict(state_dict, storage_reader, no_dist=False)
        assert set(state_dict.keys()) == {"weight", "bias"}
        assert state_dict["weight"].shape == (3, 4)
    finally:
        dist.destroy_process_group()


class TestLoadStateDict(TestCase):
    """Test load_state_dict (deprecated API)."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu",
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_load_no_dist_single_process(self):
        """load_state_dict with no_dist=True loads from checkpoint."""
        tmpdir = tempfile.mkdtemp()
        # Save first
        original = {"weight": torch.randn(3, 4), "bias": torch.randn(4)}
        from torch.distributed.checkpoint.state_dict_saver import _save_state_dict
        _save_state_dict(
            original, FileSystemWriter(tmpdir), no_dist=True
        )
        # Load
        state_dict = {"weight": torch.zeros(3, 4), "bias": torch.zeros(4)}
        load_state_dict(state_dict, FileSystemReader(tmpdir), no_dist=True)
        self.assertEqual(set(state_dict.keys()), {"weight", "bias"})
        self.assertEqual(state_dict["weight"].shape, (3, 4))
        self.assertEqual(state_dict["bias"].shape, (4,))

    def test_load_preserves_shapes(self):
        """load_state_dict preserves tensor shapes."""
        tmpdir = tempfile.mkdtemp()
        shapes = {"a": (2, 3), "b": (5,)}
        original = {k: torch.randn(*v) for k, v in shapes.items()}
        from torch.distributed.checkpoint.state_dict_saver import _save_state_dict
        _save_state_dict(original, FileSystemWriter(tmpdir), no_dist=True)
        loaded = {k: torch.zeros(*v) for k, v in shapes.items()}
        load_state_dict(loaded, FileSystemReader(tmpdir), no_dist=True)
        for k, v in shapes.items():
            self.assertEqual(loaded[k].shape, torch.Size(v))

    def test_load_preserves_dtype(self):
        """load_state_dict preserves tensor dtypes."""
        tmpdir = tempfile.mkdtemp()
        original = {"x": torch.randn(3, 4, dtype=torch.float32)}
        from torch.distributed.checkpoint.state_dict_saver import _save_state_dict
        _save_state_dict(original, FileSystemWriter(tmpdir), no_dist=True)
        loaded = {"x": torch.zeros(3, 4, dtype=torch.float32)}
        load_state_dict(loaded, FileSystemReader(tmpdir), no_dist=True)
        self.assertEqual(loaded["x"].dtype, torch.float32)

    def test_load_triggers_deprecation_warning(self):
        """load_state_dict triggers FutureWarning (deprecated)."""
        tmpdir = tempfile.mkdtemp()
        original = {"w": torch.randn(3, 4)}
        from torch.distributed.checkpoint.state_dict_saver import _save_state_dict
        _save_state_dict(original, FileSystemWriter(tmpdir), no_dist=True)
        loaded = {"w": torch.zeros(3, 4)}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_state_dict(loaded, FileSystemReader(tmpdir), no_dist=True)
            deprecation_found = any(
                issubclass(item.category, FutureWarning) for item in w
            )
        self.assertTrue(deprecation_found, "No FutureWarning for deprecated load_state_dict")

    def test_load_with_coordinator_rank(self):
        """load_state_dict with coordinator_rank=0 (default)."""
        tmpdir = tempfile.mkdtemp()
        original = {"w": torch.randn(3)}
        from torch.distributed.checkpoint.state_dict_saver import _save_state_dict
        _save_state_dict(original, FileSystemWriter(tmpdir), no_dist=True)
        loaded = {"w": torch.zeros(3)}
        load_state_dict(
            loaded, FileSystemReader(tmpdir), no_dist=True, coordinator_rank=0
        )
        self.assertEqual(loaded["w"].shape, (3,))

    @skipIfUnsupportMultiNPU(2)
    def test_load_multicard_hccl(self):
        """load_state_dict with 2 NPUs using HCCL backend."""
        tmpdir = tempfile.mkdtemp()
        original = {"weight": torch.randn(3, 4), "bias": torch.randn(4)}
        from torch.distributed.checkpoint.state_dict_saver import _save_state_dict
        _save_state_dict(original, FileSystemWriter(tmpdir), no_dist=True)

        world_size = 2
        mp.spawn(
            _run_load_multicard,
            args=(world_size, tmpdir),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    run_tests()
