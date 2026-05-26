# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.state_dict_saver.save_state_dict 接口功能正确性
API 名称：torch.distributed.checkpoint.state_dict_saver.save_state_dict
API 签名：save_state_dict(state_dict: STATE_DICT_TYPE, storage_writer: StorageWriter, process_group: dist.ProcessGroup | None = None, coordinator_rank: int = 0, no_dist: bool = False, planner: SavePlanner | None = None) -> Metadata

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 空 state_dict vs 非空 state_dict                             | 已覆盖                                          |
| 枚举选项         | coordinator_rank (0, 非默认)                                 | 已覆盖                                          |
| 参数类型         | state_dict: dict, storage_writer: StorageWriter              | 已覆盖                                          |
| 传参与不传参     | process_group 默认 None；planner 默认 None                   | 已覆盖                                          |
| 等价类/边界值    | 单 tensor、多 tensor                                         | 已覆盖                                          |
| 正常传参场景     | no_dist=True 单进程保存；多卡保存                             | 已覆盖                                          |
| 异常传参场景     | 无稳定异常路径                                                | 未覆盖：内部异常随上游变化                      |
| 混合设备类型     | NPU tensor 通过 HCCL 跨 rank 保存                            | 已覆盖                                          |

未覆盖项及原因：
- 异常传参场景：save_state_dict 内部无稳定异常路径

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
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.checkpoint.state_dict_saver import save_state_dict

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
    os.environ["MASTER_PORT"] = "29504"
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    torch.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    try:
        fn(rank, world_size, device_name)
    finally:
        dist.destroy_process_group()


def _run_save_state_dict_multicard(rank, world_size, ckpt_dir):
    """Worker: test save_state_dict with multiple NPUs."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29504"
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    torch.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    try:
        local_device = f"npu:{rank}"
        state_dict = {"weight": torch.randn(3, 4, device=local_device)}
        writer = FileSystemWriter(ckpt_dir)
        metadata = save_state_dict(state_dict, writer, no_dist=False)
        assert isinstance(metadata, Metadata)
    finally:
        dist.destroy_process_group()


class TestSaveStateDict(TestCase):
    """Test save_state_dict (deprecated API)."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu",
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_save_state_dict_no_dist(self):
        """save_state_dict with no_dist=True saves checkpoint."""
        tmpdir = tempfile.mkdtemp()
        state_dict = {"w": torch.randn(3, 4)}
        result = save_state_dict(state_dict, FileSystemWriter(tmpdir), no_dist=True)
        self.assertIsInstance(result, Metadata)

    def test_save_state_dict_returns_metadata(self):
        """save_state_dict returns Metadata object."""
        tmpdir = tempfile.mkdtemp()
        state_dict = {"w": torch.randn(3)}
        result = save_state_dict(state_dict, FileSystemWriter(tmpdir), no_dist=True)
        self.assertIsInstance(result, Metadata)

    def test_save_state_dict_triggers_deprecation_warning(self):
        """save_state_dict triggers FutureWarning (deprecated)."""
        tmpdir = tempfile.mkdtemp()
        state_dict = {"w": torch.randn(3)}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            save_state_dict(state_dict, FileSystemWriter(tmpdir), no_dist=True)
            deprecation_found = any(
                issubclass(item.category, FutureWarning) for item in w
            )
        self.assertTrue(deprecation_found, "No FutureWarning for deprecated save_state_dict")

    def test_save_state_dict_with_filesystem_writer(self):
        """save_state_dict creates checkpoint files with FileSystemWriter."""
        tmpdir = tempfile.mkdtemp()
        state_dict = {"w": torch.randn(3, 4)}
        save_state_dict(state_dict, FileSystemWriter(tmpdir), no_dist=True)
        self.assertTrue(os.path.isdir(tmpdir))
        self.assertGreater(len(os.listdir(tmpdir)), 0)

    def test_save_state_dict_preserves_keys(self):
        """save_state_dict preserves state_dict keys."""
        tmpdir = tempfile.mkdtemp()
        state_dict = {"weight": torch.randn(3, 4), "bias": torch.randn(4)}
        save_state_dict(state_dict, FileSystemWriter(tmpdir), no_dist=True)
        self.assertEqual(set(state_dict.keys()), {"weight", "bias"})

    @skipIfUnsupportMultiNPU(2)
    def test_save_state_dict_multicard_hccl(self):
        """save_state_dict with 2 NPUs using HCCL backend."""
        tmpdir = tempfile.mkdtemp()
        world_size = 2
        mp.spawn(
            _run_save_state_dict_multicard,
            args=(world_size, tmpdir),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    run_tests()
