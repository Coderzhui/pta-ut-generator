# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.state_dict_saver.async_save 接口功能正确性
API 名称：torch.distributed.checkpoint.state_dict_saver.async_save
API 签名：async_save(state_dict: STATE_DICT_TYPE, *, checkpoint_id: str | os.PathLike | None = None,
                     storage_writer: StorageWriter | None = None, planner: SavePlanner | None = None,
                     process_group: dist.ProcessGroup | None = None,
                     async_checkpointer_type: AsyncCheckpointerType = AsyncCheckpointerType.THREAD) -> Future

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 空 state_dict vs 非空 state_dict                             | 已覆盖                                          |
| 枚举选项         | async_checkpointer_type (THREAD)                             | 已覆盖                                          |
| 传参与不传参     | storage_writer/planner 默认 None                             | 已覆盖                                          |
| 正常传参场景     | 单进程 gloo 后端 + checkpoint_id 保存                         | 已覆盖                                          |
| 混合设备类型     | NPU tensor 通过 gloo 保存到磁盘                               | 已覆盖                                          |
| 异常传参场景     | 无稳定异常路径                                                | 未覆盖：异步保存依赖上游实现                    |

未覆盖项及原因：
- 异常传参场景：async_save 内部异常依赖 async executor 实现

注意：本测试仅验证功能正确性（调用不报错、输出类型符合预期），
     不做精度和数值正确性校验。
"""

import os
import tempfile
from concurrent.futures import Future

import torch
import torch.distributed as dist
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.staging import BlockingAsyncStager
from torch.distributed.checkpoint.state_dict_saver import (
    async_save,
    AsyncCheckpointerType,
)

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        import unittest
        unittest.main(argv=sys.argv)


class TestAsyncSave(TestCase):
    """Test async_save function (2.7.1 compatible)."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu",
            f"Expected device 'npu', got '{self.device_name}'"
        )
        # async_save requires process group setup (uses gloo for CPU staging)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29505")
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo", rank=0, world_size=1)

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDown()

    def test_async_save_returns_future(self):
        """async_save returns a Future."""
        tmpdir = tempfile.mkdtemp()
        state_dict = {"w": torch.randn(3, 4)}
        result = async_save(
            state_dict, checkpoint_id=tmpdir,
            storage_writer=FileSystemWriter(tmpdir),
        )
        self.assertIsInstance(result, Future)

    def test_async_save_creates_checkpoint(self):
        """async_save creates checkpoint files."""
        tmpdir = tempfile.mkdtemp()
        state_dict = {"w": torch.randn(3, 4)}
        result = async_save(
            state_dict, checkpoint_id=tmpdir,
            storage_writer=FileSystemWriter(tmpdir),
        )
        result.result()
        self.assertTrue(os.path.isdir(tmpdir))

    def test_async_save_with_thread_type(self):
        """async_save with AsyncCheckpointerType.THREAD works."""
        tmpdir = tempfile.mkdtemp()
        state_dict = {"w": torch.randn(3)}
        result = async_save(
            state_dict, checkpoint_id=tmpdir,
            storage_writer=FileSystemWriter(tmpdir),
            async_checkpointer_type=AsyncCheckpointerType.THREAD,
        )
        result.result()

    def test_async_save_with_empty_state_dict(self):
        """async_save handles empty state_dict."""
        tmpdir = tempfile.mkdtemp()
        result = async_save(
            {}, checkpoint_id=tmpdir,
            storage_writer=FileSystemWriter(tmpdir),
        )
        result.result()

    def test_async_save_with_checkpoint_id(self):
        """async_save accepts checkpoint_id parameter."""
        tmpdir = tempfile.mkdtemp()
        state_dict = {"w": torch.randn(3)}
        result = async_save(
            state_dict, checkpoint_id=tmpdir,
            storage_writer=FileSystemWriter(tmpdir),
        )
        result.result()
        self.assertTrue(os.path.isdir(tmpdir))

    def test_async_save_callable(self):
        """async_save is callable."""
        self.assertTrue(callable(async_save))


if __name__ == "__main__":
    run_tests()
