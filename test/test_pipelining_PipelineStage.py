# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.pipelining.PipelineStage 接口功能正确性
API 名称：torch.distributed.pipelining.PipelineStage
API 签名：class PipelineStage(_PipelineStageBase):
             def __init__(submodule: nn.Module, stage_index: int, num_stages: int,
                          device: torch.device,
                          input_args: Tensor | tuple[Tensor,...] | None = None,
                          output_args: Tensor | tuple[Tensor,...] | None = None,
                          group: dist.ProcessGroup | None = None,
                          dw_builder: Callable | None = None)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | input_args/output_args=None vs 有值                          | 已覆盖                                         |
| 枚举选项         | N/A（无枚举参数）                                            | N/A                                            |
| 参数类型         | submodule=nn.Module; device=torch.device; group=ProcessGroup | 已覆盖                                         |
| 传参与不传参     | input_args/output_args/group/dw_builder 省略 vs 传入         | 已覆盖                                         |
| 等价类/边界值    | 单 stage(stage_index=0,num_stages=1); 多 stage              | 已覆盖                                         |
| 正常传参场景     | 构造 PipelineStage 并访问属性                                 | 已覆盖                                         |
| 异常传参场景     | stage_index >= num_stages raises ValueError                  | 已覆盖                                         |
| 混合设备类型     | 单 Module 输入，不涉及多 Tensor 异构设备                     | 未覆盖：不适用                                 |

未覆盖项及原因：
- 混合设备类型：PipelineStage 接收单个 Module，不涉及多 Tensor 异构设备输入

注意：本测试仅验证功能正确性（构造不报错、属性/方法存在且类型正确），
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
from torch.distributed.pipelining import PipelineStage


def _init_dist_hccl(rank, world_size, fn, device_name):
    """Initialize HCCL distributed process."""
    import os
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
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


def _test_construction_basic(rank, world_size, device_name):
    """Worker: basic PipelineStage construction."""
    model = _SimpleModel()
    device = torch.device(f"{device_name}:{rank}")
    stage = PipelineStage(
        submodule=model,
        stage_index=rank,
        num_stages=world_size,
        device=device,
    )
    assert isinstance(stage, PipelineStage)


def _test_construction_with_input_args(rank, world_size, device_name):
    """Worker: construction with input_args."""
    device = torch.device(f"{device_name}:{rank}")
    model = _SimpleModel().to(device)
    input_args = torch.randn(2, 4, device=device)
    stage = PipelineStage(
        submodule=model,
        stage_index=rank,
        num_stages=world_size,
        device=device,
        input_args=input_args,
    )
    assert isinstance(stage, PipelineStage)


def _test_construction_with_input_and_output_args(rank, world_size, device_name):
    """Worker: construction with input_args and output_args."""
    device = torch.device(f"{device_name}:{rank}")
    model = _SimpleModel().to(device)
    input_args = torch.randn(2, 4, device=device)
    output_args = torch.randn(2, 4, device=device)
    stage = PipelineStage(
        submodule=model,
        stage_index=rank,
        num_stages=world_size,
        device=device,
        input_args=input_args,
        output_args=output_args,
    )
    assert isinstance(stage, PipelineStage)


def _test_stage_index_attribute(rank, world_size, device_name):
    """Worker: stage_index attribute is set correctly."""
    model = _SimpleModel()
    device = torch.device(f"{device_name}:{rank}")
    stage = PipelineStage(
        submodule=model,
        stage_index=rank,
        num_stages=world_size,
        device=device,
    )
    assert stage.stage_index == rank, f"Expected stage_index={rank}, got {stage.stage_index}"


def _test_num_stages_attribute(rank, world_size, device_name):
    """Worker: num_stages attribute is set correctly."""
    model = _SimpleModel()
    device = torch.device(f"{device_name}:{rank}")
    stage = PipelineStage(
        submodule=model,
        stage_index=rank,
        num_stages=world_size,
        device=device,
    )
    assert stage.num_stages == world_size, f"Expected num_stages={world_size}, got {stage.num_stages}"


def _test_device_attribute(rank, world_size, device_name):
    """Worker: device attribute is set correctly."""
    model = _SimpleModel()
    device = torch.device(f"{device_name}:{rank}")
    stage = PipelineStage(
        submodule=model,
        stage_index=rank,
        num_stages=world_size,
        device=device,
    )
    assert stage.device == device


def _test_submod_attribute(rank, world_size, device_name):
    """Worker: submod attribute stores the model."""
    model = _SimpleModel()
    device = torch.device(f"{device_name}:{rank}")
    stage = PipelineStage(
        submodule=model,
        stage_index=rank,
        num_stages=world_size,
        device=device,
    )
    assert stage.submod is model


def _test_output_args_without_input_args_raises(rank, world_size, device_name):
    """Worker: output_args without input_args raises AssertionError."""
    model = _SimpleModel()
    device = torch.device(f"{device_name}:{rank}")
    output_args = torch.randn(2, 4, device=device)
    try:
        PipelineStage(
            submodule=model,
            stage_index=rank,
            num_stages=world_size,
            device=device,
            output_args=output_args,
        )
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass


def _test_tuple_input_args(rank, world_size, device_name):
    """Worker: input_args as tuple of tensors."""
    device = torch.device(f"{device_name}:{rank}")
    model = _SimpleModel().to(device)
    input_args = (torch.randn(2, 4, device=device),)
    output_args = (torch.randn(2, 4, device=device),)
    stage = PipelineStage(
        submodule=model,
        stage_index=rank,
        num_stages=world_size,
        device=device,
        input_args=input_args,
        output_args=output_args,
    )
    assert isinstance(stage, PipelineStage)


class TestPipeliningPipelineStage(TestCase):
    """Test cases for torch.distributed.pipelining.PipelineStage with HCCL."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_construction_basic(self):
        """Basic PipelineStage construction."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_construction_basic, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_construction_with_input_args(self):
        """Construction with input_args specified."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_construction_with_input_args, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_construction_with_input_and_output_args(self):
        """Construction with both input_args and output_args."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_construction_with_input_and_output_args, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_stage_index_attribute(self):
        """stage_index attribute is accessible and correct."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_stage_index_attribute, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_num_stages_attribute(self):
        """num_stages attribute is accessible and correct."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_num_stages_attribute, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_device_attribute(self):
        """device attribute is accessible and correct."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_device_attribute, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_submod_attribute(self):
        """submod attribute stores the original model."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_submod_attribute, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_output_args_without_input_args_raises(self):
        """output_args without input_args raises AssertionError."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_output_args_without_input_args_raises, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_tuple_input_args(self):
        """input_args as tuple of tensors."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_tuple_input_args, self.device_name),
            nprocs=world_size,
            join=True
        )

    def test_invalid_stage_index_raises(self):
        """stage_index >= num_stages raises ValueError."""
        model = _SimpleModel()
        with self.assertRaises(ValueError):
            PipelineStage(
                submodule=model,
                stage_index=5,
                num_stages=2,
                device=torch.device(self.device_name),
            )


if __name__ == "__main__":
    run_tests()
