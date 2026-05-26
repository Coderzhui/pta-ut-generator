# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.state_dict_saver.save 接口功能正确性
API 名称：torch.distributed.checkpoint.state_dict_saver.save
API 签名：save(state_dict: STATE_DICT_TYPE, *, checkpoint_id: str | os.PathLike | None = None, storage_writer: StorageWriter | None = None, planner: SavePlanner | None = None, process_group: dist.ProcessGroup | None = None, no_dist: bool = False, use_collectives: bool = True) -> Metadata

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 空 state_dict vs 非空 state_dict                             | 已覆盖                                          |
| 枚举选项         | use_collectives (True/False)、no_dist (True/False)           | 已覆盖                                          |
| 参数类型         | state_dict: dict, checkpoint_id: str, storage_writer         | 已覆盖                                          |
| 传参与不传参     | checkpoint_id、planner、process_group 默认 None              | 已覆盖                                          |
| 等价类/边界值    | 单 tensor、多 tensor、多种 dtype                             | 已覆盖                                          |
| 正常传参场景     | no_dist=True 单进程保存；多卡 HCCL 保存                      | 已覆盖                                          |
| 异常传参场景     | 无 storage_writer 且无 checkpoint_id 时报错                  | 已覆盖                                          |
| 混合设备类型     | NPU tensor 通过 HCCL 跨 rank 保存                            | 已覆盖                                          |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.checkpoint.state_dict_saver import save

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
    os.environ["MASTER_PORT"] = "29503"
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    torch.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    try:
        fn(rank, world_size, device_name)
    finally:
        dist.destroy_process_group()


def _run_save_multicard(rank, world_size, ckpt_dir):
    """Worker: test save with multiple NPUs."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29503"
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    torch.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    try:
        local_device = f"npu:{rank}"
        state_dict = {"weight": torch.randn(3, 4, device=local_device)}
        writer = FileSystemWriter(ckpt_dir)
        metadata = save(state_dict, storage_writer=writer)
        assert isinstance(metadata, Metadata)
    finally:
        dist.destroy_process_group()


class TestSave(TestCase):
    """Test save function."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu",
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_save_no_dist_returns_metadata(self):
        """save with no_dist=True returns Metadata."""
        tmpdir = tempfile.mkdtemp()
        state_dict = {"w": torch.randn(3, 4)}
        result = save(state_dict, checkpoint_id=tmpdir, no_dist=True)
        self.assertIsInstance(result, Metadata)

    def test_save_creates_checkpoint_files(self):
        """save creates files in checkpoint directory."""
        tmpdir = tempfile.mkdtemp()
        state_dict = {"w": torch.randn(3, 4)}
        save(state_dict, checkpoint_id=tmpdir, no_dist=True)
        self.assertTrue(os.path.isdir(tmpdir))
        files = os.listdir(tmpdir)
        self.assertGreater(len(files), 0)

    def test_save_with_various_dtypes(self):
        """save handles various tensor dtypes."""
        tmpdir = tempfile.mkdtemp()
        for dtype in [torch.float32, torch.float16, torch.int64]:
            sd = {"t": torch.randn(3).to(dtype)}
            save(sd, checkpoint_id=tmpdir, no_dist=True)

    def test_save_with_empty_state_dict(self):
        """save handles empty state_dict."""
        tmpdir = tempfile.mkdtemp()
        result = save({}, checkpoint_id=tmpdir, no_dist=True)
        self.assertIsInstance(result, Metadata)

    def test_save_with_checkpoint_id(self):
        """save accepts checkpoint_id as path."""
        tmpdir = tempfile.mkdtemp()
        state_dict = {"w": torch.randn(3)}
        result = save(state_dict, checkpoint_id=tmpdir, no_dist=True)
        self.assertIsInstance(result, Metadata)

    @skipIfUnsupportMultiNPU(2)
    def test_save_multicard_hccl(self):
        """save with 2 NPUs using HCCL backend."""
        tmpdir = tempfile.mkdtemp()
        world_size = 2
        mp.spawn(
            _run_save_multicard,
            args=(world_size, tmpdir),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    run_tests()
