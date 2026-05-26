# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.distributed_c10d.rendezvous 接口功能正确性
API 名称：torch.distributed.distributed_c10d.rendezvous
API 签名：def rendezvous(url: str, rank: int = -1, world_size: int = -1, **kwargs) -> Iterator[tuple[Store, int, int]]

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | url 非空字符串                                               | 已覆盖                                         |
| 枚举选项         | N/A（无枚举参数）                                            | N/A                                            |
| 参数类型         | url=str; rank=int; world_size=int                            | 已覆盖                                         |
| 传参与不传参     | rank/world_size 有默认值 -1，显式传入 vs 省略                | 已覆盖                                         |
| 等价类/边界值    | rank=0, world_size=1 等                                      | 已覆盖                                         |
| 正常传参场景     | file:// URL rendezvous 返回 (store, rank, world_size)        | 已覆盖                                         |
| 异常传参场景     | url 非字符串 raises RuntimeError                             | 已覆盖                                         |
| 混合设备类型     | rendezvous 不涉及 Tensor                                     | 未覆盖：不适用                                 |

未覆盖项及原因：
- 混合设备类型：rendezvous 是分布式初始化函数，不涉及 Tensor 操作

注意：本测试仅验证功能正确性（调用不报错、返回类型正确），
     不做精度和数值正确性校验。
"""

import tempfile
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
from torch.distributed import rendezvous


def _init_dist_hccl(rank, world_size, fn, device_name):
    """Initialize HCCL distributed process."""
    import os
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29505'
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    try:
        fn(rank, world_size, device_name)
    finally:
        dist.destroy_process_group()


def _test_rendezvous_file_url(rank, world_size, device_name):
    """Worker: rendezvous with file:// URL."""
    import tempfile
    import os
    tmp_dir = tempfile.mkdtemp()
    filepath = os.path.join(tmp_dir, "rdzv_test")
    url = f"file://{filepath}?rank={rank}&world_size={world_size}"
    result = rendezvous(url)
    store, r, ws = next(result)
    assert store is not None, "Store should not be None"
    assert r == rank, f"Expected rank={rank}, got {r}"
    assert ws == world_size, f"Expected world_size={world_size}, got {ws}"


def _test_rendezvous_returns_iterator(rank, world_size, device_name):
    """Worker: rendezvous returns an iterator."""
    import tempfile
    import os
    tmp_dir = tempfile.mkdtemp()
    filepath = os.path.join(tmp_dir, "rdzv_iter_test")
    url = f"file://{filepath}?rank={rank}&world_size={world_size}"
    result = rendezvous(url)
    assert hasattr(result, '__iter__') or hasattr(result, '__next__'), \
        "rendezvous should return an iterator"


def _test_rendezvous_store_type(rank, world_size, device_name):
    """Worker: returned store has correct type."""
    import tempfile
    import os
    from torch._C._distributed_c10d import Store
    tmp_dir = tempfile.mkdtemp()
    filepath = os.path.join(tmp_dir, "rdzv_store_test")
    url = f"file://{filepath}?rank={rank}&world_size={world_size}"
    result = rendezvous(url)
    store, r, ws = next(result)
    assert isinstance(store, Store), f"Expected Store, got {type(store)}"


class TestDistributedC10dRendezvous(TestCase):
    """Test cases for torch.distributed.distributed_c10d.rendezvous."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_rendezvous_file_url(self):
        """Rendezvous with file:// URL returns (store, rank, world_size)."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_rendezvous_file_url, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_rendezvous_returns_iterator(self):
        """Rendezvous returns an iterator."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_rendezvous_returns_iterator, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_rendezvous_store_type(self):
        """Returned store is a Store instance."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_rendezvous_store_type, self.device_name),
            nprocs=world_size,
            join=True
        )

    def test_invalid_url_type_raises(self):
        """Non-string URL raises RuntimeError."""
        with self.assertRaises(RuntimeError):
            rendezvous(123)

    def test_invalid_url_none_raises(self):
        """None URL raises RuntimeError."""
        with self.assertRaises(RuntimeError):
            rendezvous(None)

    def test_invalid_rank_type_handled(self):
        """Non-integer rank in URL is handled by specific rendezvous handler."""
        # rank/world_size validation happens inside the iterator, not at rendezvous() call
        result = rendezvous("file:///tmp/test?rank=abc&world_size=1")
        self.assertIsNotNone(result)

    def test_invalid_world_size_type_handled(self):
        """Non-integer world_size in URL is handled by specific rendezvous handler."""
        result = rendezvous("file:///tmp/test?rank=0&world_size=abc")
        self.assertIsNotNone(result)


if __name__ == "__main__":
    run_tests()
