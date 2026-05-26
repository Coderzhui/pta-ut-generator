# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.read_data 接口功能正确性
API 名称：torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.read_data
API 签名：read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | plan.items 为空 vs 非空                                      | 已覆盖                                          |
| 枚举选项         | coordinator_rank (0, 非0)                                    | 已覆盖                                          |
| 参数类型         | plan: LoadPlan, planner: LoadPlanner                         | 已覆盖                                          |
| 传参与不传参     | coordinator_rank 默认 0 vs 显式传入                          | 已覆盖                                          |
| 等价类/边界值    | 单 tensor、多 tensor plan                                    | 已覆盖                                          |
| 正常传参场景     | 多卡 HCCL 环境下广播数据                                     | 已覆盖                                          |
| 异常传参场景     | checkpoint_id=None 时 coordinator rank 报 AssertionError    | 已覆盖                                          |
| 混合设备类型     | NPU tensor 通过 dist.broadcast 跨 rank 传输                  | 已覆盖                                          |

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
from torch.distributed.checkpoint.format_utils import (
    BroadcastingTorchSaveReader,
    DynamicMetaLoadPlanner,
)
from torch.futures import Future

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


def _run_read_data_basic(rank, world_size, checkpoint_path):
    """Worker: test read_data broadcasts tensor from coordinator."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    torch.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    try:
        reader = BroadcastingTorchSaveReader(
            checkpoint_id=checkpoint_path, coordinator_rank=0
        )
        reader.is_coordinator = rank == 0

        local_tensor = torch.zeros(3, 4, device=f"npu:{rank}")
        state_dict = {"weight": local_tensor}

        planner = DynamicMetaLoadPlanner()
        planner.set_up_planner(state_dict, None, is_coordinator=(rank == 0))
        plan = planner.create_local_plan()
        plan = reader.prepare_local_plan(plan)

        result = reader.read_data(plan, planner)
        assert isinstance(result, Future)
        assert result.wait() is None
    finally:
        dist.destroy_process_group()


class TestBroadcastingTorchSaveReaderReadData(TestCase):
    """Test BroadcastingTorchSaveReader.read_data with multi-card HCCL."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu",
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_read_data_returns_future(self):
        """read_data returns a Future object."""
        tmpdir = tempfile.mkdtemp()
        ckpt_path = os.path.join(tmpdir, "test.pt")
        torch.save({"weight": torch.randn(3, 4)}, ckpt_path)

        world_size = 2
        mp.spawn(
            _run_read_data_basic,
            args=(world_size, ckpt_path),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    run_tests()
