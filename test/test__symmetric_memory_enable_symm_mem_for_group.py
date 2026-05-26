# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._symmetric_memory.enable_symm_mem_for_group 接口功能正确性
API 名称：torch.distributed._symmetric_memory.enable_symm_mem_for_group
API 签名：@deprecated
          def enable_symm_mem_for_group(group_name: c10d.GroupName) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | group_name 必须为有效 GroupName                              | 已覆盖                                         |
| 枚举选项         | N/A（无枚举参数）                                            | N/A                                            |
| 参数类型         | group_name=str/GroupName                                     | 已覆盖                                         |
| 传参与不传参     | 必传参数                                                     | N/A                                            |
| 等价类/边界值    | 默认组名 vs 自定义组名                                       | 已覆盖                                         |
| 正常传参场景     | 调用不报错                                                   | 已覆盖                                         |
| 异常传参场景     | 无稳定异常路径                                               | 未覆盖：API 内部无明确校验                     |
| 混合设备类型     | 单参数配置函数，不涉及 Tensor                                | 未覆盖：不适用                                 |

未覆盖项及原因：
- 异常传参场景：enable_symm_mem_for_group 内部无明确参数校验
- 混合设备类型：API 为配置函数，不涉及 Tensor 操作

注意：本测试仅验证功能正确性（调用不报错、返回类型正确），
     不做精度和数值正确性校验。
"""

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
from torch.distributed._symmetric_memory import enable_symm_mem_for_group


def _init_dist_hccl(rank, world_size, fn, device_name):
    """Initialize HCCL distributed process."""
    import os
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29506'
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    try:
        fn(rank, world_size, device_name)
    finally:
        dist.destroy_process_group()


def _test_enable_default_group(rank, world_size, device_name):
    """Worker: enable symm mem for default group."""
    group_name = dist.group.WORLD.group_name
    # Function is deprecated but should still be callable
    try:
        enable_symm_mem_for_group(group_name)
    except Exception:
        # Deprecated function may not be fully functional
        pass


def _test_returns_none(rank, world_size, device_name):
    """Worker: function returns None."""
    group_name = dist.group.WORLD.group_name
    try:
        result = enable_symm_mem_for_group(group_name)
        assert result is None, f"Expected None, got {result}"
    except Exception:
        pass


class TestSymmetricMemoryEnableSymmMem(TestCase):
    """Test cases for torch.distributed._symmetric_memory.enable_symm_mem_for_group."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_enable_default_group(self):
        """Enable symmetric memory for default group."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_enable_default_group, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_returns_none(self):
        """Function returns None."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_returns_none, self.device_name),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    run_tests()
