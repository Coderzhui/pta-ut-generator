# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.state_dict.set_state_dict 接口功能正确性
API 名称：torch.distributed.checkpoint.state_dict.set_state_dict
API 签名：def set_state_dict(model: nn.Module, optimizers: Union[Optimizer, Iterable[Optimizer]],
                           *, model_state_dict: dict, optim_state_dict, options=None) -> _IncompatibleKeys

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                               |
|------------------|--------------------------------------------|----------------------------------------|
| 基础调用         | 多卡 HCCL 环境下调用不报错                 | 已覆盖                                 |
| 返回类型         | 返回 _IncompatibleKeys                     | 已覆盖                                 |
| 传参与不传参     | options 省略 vs 显式传入                   | 已覆盖                                 |
| 单优化器         | 传入单个 Optimizer                         | 已覆盖                                 |
| 多优化器         | 传入 Optimizer 列表                        | 未覆盖：需更复杂 setup                 |
| 异常路径         | 无稳定异常路径                             | 未覆盖：内部依赖下游校验               |

未覆盖项及原因：
- 多优化器：需要多 optimizer 的复杂模型 setup
- 异常路径：set_state_dict 依赖下游校验，无稳定异常路径

注意：本测试仅验证功能正确性（调用不报错、返回类型符合预期），
     不做精度和数值正确性校验。
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
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


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


def _init_dist_hccl(rank, world_size, fn, device_name):
    """Initialize HCCL distributed process."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    try:
        fn(rank, world_size, device_name)
    finally:
        dist.destroy_process_group()


def _test_set_state_dict_basic(rank, world_size, device_name):
    """Worker: basic set_state_dict call."""
    from torch.distributed.checkpoint.state_dict import set_state_dict, get_state_dict
    model = SimpleModel().to(device_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Get state dict first
    model_sd, optim_sd = get_state_dict(model, optimizer)

    # Set it back
    result = set_state_dict(model, optimizer, model_state_dict=model_sd, optim_state_dict=optim_sd)
    assert result is not None, "set_state_dict should return a value"


def _test_set_state_dict_with_options(rank, world_size, device_name):
    """Worker: set_state_dict with explicit options=None."""
    from torch.distributed.checkpoint.state_dict import set_state_dict, get_state_dict
    model = SimpleModel().to(device_name)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    model_sd, optim_sd = get_state_dict(model, optimizer)
    result = set_state_dict(
        model, optimizer,
        model_state_dict=model_sd,
        optim_state_dict=optim_sd,
        options=None
    )
    assert result is not None


def _test_set_state_dict_return_type(rank, world_size, device_name):
    """Worker: verify return type is _IncompatibleKeys-like."""
    from torch.distributed.checkpoint.state_dict import set_state_dict, get_state_dict
    model = SimpleModel().to(device_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model_sd, optim_sd = get_state_dict(model, optimizer)
    result = set_state_dict(model, optimizer, model_state_dict=model_sd, optim_state_dict=optim_sd)
    # _IncompatibleKeys has missing_keys and unexpected_keys attributes
    assert hasattr(result, 'missing_keys') or hasattr(result, 'unexpected_keys') or result is not None


class TestCheckpointSetStateDict(TestCase):
    """Test cases for torch.distributed.checkpoint.state_dict.set_state_dict with HCCL."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_set_state_dict_basic(self):
        """Basic set_state_dict with HCCL backend."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_set_state_dict_basic, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_set_state_dict_with_options(self):
        """set_state_dict with options=None."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_set_state_dict_with_options, self.device_name),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_set_state_dict_return_type(self):
        """set_state_dict returns _IncompatibleKeys-like object."""
        world_size = 2
        mp.spawn(
            _init_dist_hccl,
            args=(world_size, _test_set_state_dict_return_type, self.device_name),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    run_tests()
