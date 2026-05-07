# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.Work.wait
API 名称：torch.distributed.Work.wait
API 签名：wait() -> None（完成异步操作）

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | N/A                                                          | N/A                                            |
| 枚举选项         | N/A                                                          | N/A                                            |
| 参数类型         | N/A                                                          | N/A                                            |
| 传参与不传参     | N/A                                                          | N/A                                            |
| 等价类/边界值    | all_reduce / broadcast 异步                                  | 已覆盖                                         |
| 正常传参场景     | wait() 不抛错；is_completed 为 True（若存在）                | 已覆盖                                         |
| 异常传参场景     | N/A                                                          | 未覆盖                                         |

未覆盖项及原因：
- 无

注意：无数值校验。
"""

import os
import tempfile
import unittest

import torch
import torch.distributed as dist

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


class TestWorkWait(TestCase):
    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_wait_after_all_reduce_async(self):
        path = _init_gloo_file()
        try:
            t = torch.ones(5, dtype=torch.float32)
            work = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
            self.assertIsNotNone(work)
            work.wait()
            if hasattr(work, "is_completed"):
                self.assertTrue(work.is_completed())
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
            try:
                os.unlink(path)
            except OSError:
                pass

    def test_wait_after_broadcast_async(self):
        path = _init_gloo_file()
        try:
            t = torch.zeros(2, dtype=torch.float32)
            work = dist.broadcast(t, src=0, async_op=True)
            self.assertIsNotNone(work)
            work.wait()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
            try:
                os.unlink(path)
            except OSError:
                pass


if __name__ == "__main__":
    run_tests()
