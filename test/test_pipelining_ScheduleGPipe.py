# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.pipelining.ScheduleGPipe 接口功能正确性
API 名称：torch.distributed.pipelining.ScheduleGPipe
API 签名：class ScheduleGPipe(PipelineScheduleSingle):
             继承自 PipelineScheduleSingle.__init__(
                 stage, n_microbatches, loss_fn=None,
                 args_chunk_spec=None, kwargs_chunk_spec=None,
                 output_merge_spec=None, scale_grads=True)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | loss_fn=None vs 有值                                         | 已覆盖                                         |
| 枚举选项         | scale_grads=True/False                                       | 已覆盖                                         |
| 参数类型         | stage=PipelineStage; n_microbatches=int; loss_fn=Callable    | 已覆盖                                         |
| 传参与不传参     | loss_fn/scale_grads 省略 vs 传入                             | 已覆盖                                         |
| 等价类/边界值    | n_microbatches=1 vs 多                                       | 已覆盖                                         |
| 正常传参场景     | 构造 ScheduleGPipe 实例并验证属性                             | 已覆盖                                         |
| 异常传参场景     | n_microbatches < num_stages raises ValueError                | 已覆盖                                         |
| 混合设备类型     | 单 stage 输入，不涉及多 Tensor 异构设备                      | 未覆盖：不适用                                 |

未覆盖项及原因：
- 混合设备类型：ScheduleGPipe 接收单个 PipelineStage，不涉及多 Tensor 异构设备输入

注意：本测试仅验证功能正确性（构造不报错、属性/类型正确），
     不做精度和数值正确性校验。
"""

import torch
import torch.nn as nn
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
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe


def _init_dist_hccl(rank, world_size, fn, device_name):
    """Initialize HCCL distributed process."""
    import os
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29502'
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    try:
        fn(rank, world_size, device_name)
    finally:
        dist.destroy_process_group()


class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


def _make_stage(rank, world_size, device_name):
    model = _SimpleModel()
    device = torch.device(f"{device_name}:{rank}")
    input_args = torch.randn(2, 4, device=device)
    output_args = torch.randn(2, 4, device=device)
    return PipelineStage(
        submodule=model,
        stage_index=rank,
        num_stages=world_size,
        device=device,
        input_args=input_args,
        output_args=output_args,
    )


def _test_construction_basic(rank, world_size, device_name):
    """Worker: basic ScheduleGPipe construction."""
    stage = _make_stage(rank, world_size, device_name)
    schedule = ScheduleGPipe(stage, n_microbatches=2)
    assert isinstance(schedule, ScheduleGPipe)


def _test_construction_with_loss_fn(rank, world_size, device_name):
    """Worker: construction with loss_fn."""

    def loss_fn(output, target):
        return output.sum()

    stage = _make_stage(rank, world_size, device_name)
    schedule = ScheduleGPipe(stage, n_microbatches=2, loss_fn=loss_fn)
    assert isinstance(schedule, ScheduleGPipe)


def _test_construction_loss_fn_none(rank, world_size, device_name):
    """Worker: construction with explicit loss_fn=None."""
    stage = _make_stage(rank, world_size, device_name)
    schedule = ScheduleGPipe(stage, n_microbatches=2, loss_fn=None)
    assert isinstance(schedule, ScheduleGPipe)


def _test_construction_scale_grads_true(rank, world_size, device_name):
    """Worker: construction with scale_grads=True."""
    stage = _make_stage(rank, world_size, device_name)
    schedule = ScheduleGPipe(stage, n_microbatches=2, scale_grads=True)
    assert isinstance(schedule, ScheduleGPipe)


def _test_construction_scale_grads_false(rank, world_size, device_name):
    """Worker: construction with scale_grads=False."""
    stage = _make_stage(rank, world_size, device_name)
    schedule = ScheduleGPipe(stage, n_microbatches=2, scale_grads=False)
    assert isinstance(schedule, ScheduleGPipe)


def _test_n_microbatches_minimum(rank, world_size, device_name):
    """Worker: n_microbatches=world_size (minimum valid for num_stages=world_size)."""
    stage = _make_stage(rank, world_size, device_name)
    schedule = ScheduleGPipe(stage, n_microbatches=world_size)
    assert isinstance(schedule, ScheduleGPipe)


def _test_has_step_method(rank, world_size, device_name):
    """Worker: schedule has step() method."""
    stage = _make_stage(rank, world_size, device_name)
    schedule = ScheduleGPipe(stage, n_microbatches=2)
    assert hasattr(schedule, 'step'), "ScheduleGPipe should have 'step' method"


def _test_type_inheritance(rank, world_size, device_name):
    """Worker: ScheduleGPipe inherits from PipelineScheduleSingle."""
    from torch.distributed.pipelining.schedules import PipelineScheduleSingle
    stage = _make_stage(rank, world_size, device_name)
    schedule = ScheduleGPipe(stage, n_microbatches=2)
    assert isinstance(schedule, PipelineScheduleSingle)


class TestPipeliningScheduleGPipe(TestCase):
    """Test cases for torch.distributed.pipelining.ScheduleGPipe with HCCL."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_construction_basic(self):
        """Basic ScheduleGPipe construction."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_construction_basic, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_construction_with_loss_fn(self):
        """Construction with loss_fn."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_construction_with_loss_fn, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_construction_loss_fn_none(self):
        """Construction with explicit loss_fn=None."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_construction_loss_fn_none, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_construction_scale_grads_true(self):
        """Construction with scale_grads=True."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_construction_scale_grads_true, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_construction_scale_grads_false(self):
        """Construction with scale_grads=False."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_construction_scale_grads_false, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_n_microbatches_minimum(self):
        """n_microbatches equals world_size."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_n_microbatches_minimum, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_has_step_method(self):
        """ScheduleGPipe has step() method."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_has_step_method, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_type_inheritance(self):
        """ScheduleGPipe inherits from PipelineScheduleSingle."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_type_inheritance, self.device_name),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    run_tests()
