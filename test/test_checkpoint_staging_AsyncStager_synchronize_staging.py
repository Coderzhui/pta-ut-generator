# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.staging.AsyncStager.synchronize_staging 接口功能正确性
API 名称：torch.distributed.checkpoint.staging.AsyncStager.synchronize_staging
API 签名：synchronize_staging(self) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | N/A（无输入参数）                                             | 未覆盖：无输入参数                              |
| 枚举选项         | N/A                                                          | 未覆盖：无枚举参数                              |
| 参数类型         | 无参数                                                       | 未覆盖：无参数                                  |
| 传参与不传参     | N/A                                                          | 未覆盖：无参数                                  |
| 等价类/边界值    | N/A                                                          | 未覆盖：无参数                                  |
| 正常传参场景     | 调用 synchronize_staging 不报错                               | 已覆盖                                          |
| 异常传参场景     | 无异常路径（默认 no-op）                                      | 已覆盖：确认无异常                              |
| 混合设备类型     | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖：无输入参数，不涉及设备迁移              |

未覆盖项及原因：
- 多个维度不适用：该 API 无输入参数，为 Protocol 级 no-op 方法
- 混合设备类型：无输入参数

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import warnings

import torch
from torch.distributed.checkpoint.staging import (
    AsyncStager,
    BlockingAsyncStager,
)

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


class TestAsyncStagerSynchronizeStaging(TestCase):
    """Test AsyncStager.synchronize_staging protocol interface."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu",
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_synchronize_staging_is_defined(self):
        """AsyncStager protocol defines synchronize_staging method."""
        self.assertTrue(hasattr(AsyncStager, "synchronize_staging"))

    def test_blocking_stager_synchronize_returns_none(self):
        """BlockingAsyncStager.synchronize_staging returns None (no-op)."""
        stager = BlockingAsyncStager()
        result = stager.synchronize_staging()
        self.assertIsNone(result)

    def test_synchronize_staging_no_exception(self):
        """Calling synchronize_staging does not raise."""
        stager = BlockingAsyncStager()
        stager.synchronize_staging()

    def test_synchronize_staging_multiple_calls(self):
        """synchronize_staging can be called multiple times safely."""
        stager = BlockingAsyncStager()
        for _ in range(5):
            result = stager.synchronize_staging()
            self.assertIsNone(result)

    def test_synchronize_after_stage(self):
        """synchronize_staging after stage operation completes without error."""
        stager = BlockingAsyncStager()
        state_dict = {"weight": torch.randn(3, 4, device=self.device_name)}
        stager.stage(state_dict)
        result = stager.synchronize_staging()
        self.assertIsNone(result)

    def test_synchronize_with_cache_enabled(self):
        """synchronize_staging works with cache_staged_state_dict=True."""
        stager = BlockingAsyncStager(cache_staged_state_dict=True)
        state_dict = {"weight": torch.randn(3, 4, device=self.device_name)}
        stager.stage(state_dict)
        result = stager.synchronize_staging()
        self.assertIsNone(result)

    def test_custom_stager_synchronize(self):
        """Custom AsyncStager can implement synchronize_staging."""

        class MyStager(AsyncStager):
            _synchronize_after_execute = True
            sync_called = False

            def stage(self, state_dict):
                return state_dict

            def synchronize_staging(self):
                self.sync_called = True

            def close(self):
                pass

        stager = MyStager()
        stager.synchronize_staging()
        self.assertTrue(stager.sync_called)


if __name__ == "__main__":
    run_tests()
