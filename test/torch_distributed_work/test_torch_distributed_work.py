# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.Work 异步 collective 返回类型
API 名称：torch.distributed.Work
API 签名：C++ 绑定类型；async_op=True 时由 collective 返回

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 同步路径可能返回 None                                        | 已覆盖 async 非 None                           |
| 枚举选项         | ReduceOp.SUM                                                 | 已覆盖                                         |
| 参数类型         | Tensor on CPU(gloo)                                          | 已覆盖                                         |
| 传参与不传参     | async_op True                                                | 已覆盖                                         |
| 等价类/边界值    | 1D tensor                                                    | 已覆盖                                         |
| 正常传参场景     | isinstance(work, Work)                                       | 已覆盖                                         |
| 异常传参场景     | N/A                                                          | 未覆盖                                         |

未覆盖项及原因：
- HCCL 多卡：需 skipIfUnsupportMultiNPU，本文件单进程 gloo 基线

注意：无数值校验。
"""

import os
import tempfile
import unittest

import torch
import torch.distributed as dist
from torch.distributed import Work

try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    pass

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


def _init_gloo_file():
    if not dist.is_available():
        raise unittest.SkipTest("distributed not available")
    fd, path = tempfile.mkstemp()
    os.close(fd)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{path}",
        rank=0,
        world_size=1,
    )
    return path


class TestDistributedWork(TestCase):
    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_all_reduce_async_returns_work(self):
        path = _init_gloo_file()
        try:
            t = torch.ones(4, dtype=torch.float32)
            work = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
            self.assertIsNotNone(work)
            self.assertIsInstance(work, Work)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
            try:
                os.unlink(path)
            except OSError:
                pass

    def test_broadcast_async_returns_work(self):
        path = _init_gloo_file()
        try:
            t = torch.arange(3, dtype=torch.float32)
            work = dist.broadcast(t, src=0, async_op=True)
            self.assertIsNotNone(work)
            self.assertIsInstance(work, Work)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
            try:
                os.unlink(path)
            except OSError:
                pass


if __name__ == "__main__":
    run_tests()
