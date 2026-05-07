# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.optim.SGD.add_param_group 接口功能正确性
API 名称：torch.optim.SGD.add_param_group
API 签名：torch.optim.SGD.add_param_group(param_group)

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 添加参数组不报错                    | 已覆盖                  |
| 参数组内容       | 含 params 和自定义 lr               | 已覆盖                  |
| 多次添加         | 连续添加多个参数组                  | 已覆盖                  |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性，不做精度和数值正确性校验。
"""
import torch
import torch_npu  # noqa: F401

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    import unittest
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


class TestOptimSGDAddParamGroup(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_add_param_group_basic(self):
        """Verify adding a param group increases group count."""
        params = [torch.randn(4, 4, device=self.device_name, requires_grad=True)]
        opt = torch.optim.SGD(params, lr=0.01)
        new_params = [torch.randn(2, 2, device=self.device_name, requires_grad=True)]
        opt.add_param_group({'params': new_params})
        self.assertEqual(len(opt.param_groups), 2)

    def test_npu_add_param_group_with_lr(self):
        """Verify new param group can have a custom learning rate."""
        params = [torch.randn(4, 4, device=self.device_name, requires_grad=True)]
        opt = torch.optim.SGD(params, lr=0.01)
        new_params = [torch.randn(2, 2, device=self.device_name, requires_grad=True)]
        opt.add_param_group({'params': new_params, 'lr': 0.001})
        self.assertEqual(opt.param_groups[1]['lr'], 0.001)

    def test_npu_add_multiple_groups(self):
        """Verify adding multiple param groups sequentially."""
        params = [torch.randn(4, 4, device=self.device_name, requires_grad=True)]
        opt = torch.optim.SGD(params, lr=0.01)
        for _ in range(3):
            opt.add_param_group({'params': [torch.randn(2, 2, device=self.device_name, requires_grad=True)]})
        self.assertEqual(len(opt.param_groups), 4)

    def test_cpu_baseline(self):
        """CPU baseline: verify add_param_group works on CPU."""
        params = [torch.randn(4, 4, requires_grad=True)]
        opt = torch.optim.SGD(params, lr=0.01)
        opt.add_param_group({'params': [torch.randn(2, 2, requires_grad=True)]})
        self.assertEqual(len(opt.param_groups), 2)


if __name__ == "__main__":
    run_tests()
