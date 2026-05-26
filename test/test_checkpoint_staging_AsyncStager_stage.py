# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.staging.AsyncStager.stage 接口功能正确性
API 名称：torch.distributed.checkpoint.staging.AsyncStager.stage
API 签名：stage(self, state_dict: STATE_DICT_TYPE) -> Future[STATE_DICT_TYPE] | STATE_DICT_TYPE

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 空 state_dict vs 非空 state_dict                             | 已覆盖：空 dict 和含 tensor 的 dict             |
| 枚举选项         | N/A（无枚举参数）                                            | 未覆盖：无枚举参数                              |
| 参数类型         | state_dict 类型为 Dict[str, Any]                             | 已覆盖：含 Tensor 的 dict                       |
| 传参与不传参     | N/A（无可选参数）                                             | 未覆盖：无可选参数                              |
| 等价类/边界值    | 空 dict、单元素 tensor、多维 tensor                           | 已覆盖                                          |
| 正常传参场景     | 传入合法 state_dict，验证返回类型                             | 已覆盖                                          |
| 异常传参场景     | 调用 Protocol 默认实现触发 NotImplementedError                 | 已覆盖                                          |
| 混合设备类型     | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖：Protocol 接口测试不涉及设备迁移         |

未覆盖项及原因：
- 枚举选项：该 API 无枚举参数
- 传参与不传参：该 API 无可选参数
- 混合设备类型：Protocol 接口级别测试，不涉及具体设备迁移

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import torch
from torch.distributed.checkpoint.staging import (
    AsyncStager,
    BlockingAsyncStager,
)
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


class TestAsyncStagerStage(TestCase):
    """Test AsyncStager.stage protocol interface."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu",
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_async_stager_is_protocol(self):
        """AsyncStager is a runtime-checkable Protocol."""
        self.assertTrue(hasattr(AsyncStager, "stage"))

    def test_default_stage_raises_not_implemented(self):
        """Calling stage on base Protocol instance raises NotImplementedError."""
        # AsyncStager is a Protocol; direct instantiation is possible but stage
        # raises NotImplementedError by default.

        class MinimalStager(AsyncStager):
            _synchronize_after_execute = True

            def close(self):
                pass

        stager = MinimalStager()
        with self.assertRaises(NotImplementedError):
            stager.stage({})

    def test_blocking_stager_satisfies_protocol(self):
        """BlockingAsyncStager satisfies AsyncStager protocol."""
        stager = BlockingAsyncStager()
        self.assertIsInstance(stager, AsyncStager)

    def test_blocking_stager_stage_returns_dict(self):
        """BlockingAsyncStager.stage returns STATE_DICT_TYPE."""
        stager = BlockingAsyncStager()
        state_dict = {"weight": torch.randn(3, 4, device=self.device_name)}
        result = stager.stage(state_dict)
        self.assertIsInstance(result, dict)

    def test_blocking_stager_stage_preserves_keys(self):
        """BlockingAsyncStager.stage returns dict with same keys."""
        stager = BlockingAsyncStager()
        state_dict = {
            "weight": torch.randn(3, 4, device=self.device_name),
            "bias": torch.randn(4, device=self.device_name),
        }
        result = stager.stage(state_dict)
        self.assertEqual(set(result.keys()), set(state_dict.keys()))

    def test_blocking_stager_stage_output_on_cpu(self):
        """BlockingAsyncStager.stage returns tensors on CPU."""
        stager = BlockingAsyncStager()
        state_dict = {"weight": torch.randn(3, 4, device=self.device_name)}
        result = stager.stage(state_dict)
        self.assertEqual(result["weight"].device.type, "cpu")

    def test_custom_implementation_satisfies_protocol(self):
        """Custom AsyncStager subclass satisfies isinstance check."""

        class MyStager(AsyncStager):
            _synchronize_after_execute = True

            def stage(self, state_dict):
                return state_dict

            def close(self):
                pass

        stager = MyStager()
        self.assertIsInstance(stager, AsyncStager)
        result = stager.stage({"x": torch.randn(2)})
        self.assertIsInstance(result, dict)

    def test_stage_with_empty_state_dict(self):
        """BlockingAsyncStager.stage works with empty state_dict."""
        stager = BlockingAsyncStager()
        result = stager.stage({})
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)

    def test_stage_with_single_element_tensor(self):
        """BlockingAsyncStager.stage handles single-element tensors."""
        stager = BlockingAsyncStager()
        state_dict = {"scalar": torch.tensor(42.0, device=self.device_name)}
        result = stager.stage(state_dict)
        self.assertEqual(result["scalar"].shape, torch.Size([]))


if __name__ == "__main__":
    run_tests()
