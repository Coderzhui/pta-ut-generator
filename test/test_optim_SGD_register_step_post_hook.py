# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.optim.SGD.register_step_post_hook 接口功能正确性
API 名称：torch.optim.SGD.register_step_post_hook
API 签名：torch.optim.SGD.register_step_post_hook(hook) -> RemovableHook

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 注册 hook 不报错                    | 已覆盖                  |
| 返回类型         | 返回可移除对象                      | 已覆盖                  |
| hook 被调用      | step 后 hook 被执行                 | 已覆盖                  |

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


class TestOptimSGDRegisterStepPostHook(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_npu_register_hook(self):
        """Verify register_step_post_hook returns a non-null handle."""
        params = [torch.randn(4, 4, device=self.device_name, requires_grad=True)]
        opt = torch.optim.SGD(params, lr=0.01)
        handle = opt.register_step_post_hook(lambda opt, args, kwargs: None)
        self.assertIsNotNone(handle)

    def test_npu_hook_called_on_step(self):
        """Verify registered hook is called after optimizer.step()."""
        called = [False]
        def hook(opt, args, kwargs):
            called[0] = True
        params = [torch.randn(4, 4, device=self.device_name, requires_grad=True)]
        opt = torch.optim.SGD(params, lr=0.01)
        opt.register_step_post_hook(hook)
        params[0].grad = torch.randn_like(params[0])
        opt.step()
        self.assertTrue(called[0])

    def test_npu_hook_removable(self):
        """Verify hook handle can be removed without error."""
        params = [torch.randn(4, 4, device=self.device_name, requires_grad=True)]
        opt = torch.optim.SGD(params, lr=0.01)
        handle = opt.register_step_post_hook(lambda opt, args, kwargs: None)
        handle.remove()


if __name__ == "__main__":
    run_tests()
