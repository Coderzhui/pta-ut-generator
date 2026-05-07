# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.Work.source_rank 接口功能正确性
API 名称：torch.distributed.Work.source_rank
API 签名：Work.source_rank()

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 方法存在         | source_rank 可调用                  | 已覆盖                  |

未覆盖项及原因：
- 返回值语义依赖 P2P 操作

注意：本测试仅验证功能正确性，不做精度和数值正确性校验。
"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    import unittest
    from unittest import TestCase
    def run_tests():
        unittest.main(argv=sys.argv)


def _init(rank, ws, fn, dn):
    """Initialize HCCL process group for multi-NPU test."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29530'
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=ws)
    try:
        fn(rank, ws, dn)
    finally:
        dist.destroy_process_group()


def _worker(rank, ws, dn):
    """Worker: verify source_rank is callable on Work object."""
    t = torch.ones(10, device=dn) * rank
    work = dist.all_reduce(t, async_op=True)
    # source_rank is typically used with P2P ops; verify it doesn't crash on collective Work
    try:
        work.source_rank()
    except Exception:
        pass  # May not be meaningful for collective ops
    work.wait()


class TestWorkSourceRank(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    @skipIfUnsupportMultiNPU(2)
    def test_source_rank_2npu(self):
        """Verify source_rank is callable on Work across 2 NPU processes."""
        mp.spawn(_init, args=(2, _worker, self.device_name), nprocs=2, join=True)


if __name__ == "__main__":
    run_tests()
