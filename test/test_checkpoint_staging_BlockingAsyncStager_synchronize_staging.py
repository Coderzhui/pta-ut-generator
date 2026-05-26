# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.staging.BlockingAsyncStager.synchronize_staging 接口功能正确性
API 名称：torch.distributed.checkpoint.staging.BlockingAsyncStager.synchronize_staging
API 签名：synchronize_staging(self) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | N/A（无输入参数）                                             | 未覆盖：无输入参数                              |
| 枚举选项         | N/A                                                          | 未覆盖：无枚举参数                              |
| 参数类型         | 无参数                                                       | 未覆盖：无参数                                  |
| 传参与不传参     | N/A                                                          | 未覆盖：无参数                                  |
| 等价类/边界值    | N/A                                                          | 未覆盖：无参数                                  |
| 正常传参场景     | 调用不报错，返回 None                                         | 已覆盖                                          |
| 异常传参场景     | 无异常路径（no-op 方法）                                      | 已覆盖：确认不抛异常                            |
| 混合设备类型     | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖：无输入参数，不涉及设备                  |

未覆盖项及原因：
- 多个维度不适用：该方法为 no-op，无输入参数
- 混合设备类型：不涉及设备操作

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import torch
from torch.distributed.checkpoint.staging import BlockingAsyncStager

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


class TestBlockingAsyncStagerSynchronizeStaging(TestCase):
    """Test BlockingAsyncStager.synchronize_staging (no-op)."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu",
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_synchronize_returns_none(self):
        """synchronize_staging returns None."""
        stager = BlockingAsyncStager()
        result = stager.synchronize_staging()
        self.assertIsNone(result)

    def test_synchronize_no_exception(self):
        """synchronize_staging does not raise."""
        stager = BlockingAsyncStager()
        stager.synchronize_staging()

    def test_synchronize_multiple_calls(self):
        """synchronize_staging can be called multiple times."""
        stager = BlockingAsyncStager()
        for _ in range(10):
            self.assertIsNone(stager.synchronize_staging())

    def test_synchronize_after_stage(self):
        """synchronize_staging after stage is a valid sequence."""
        stager = BlockingAsyncStager()
        state_dict = {"w": torch.randn(3, 4, device=self.device_name)}
        stager.stage(state_dict)
        self.assertIsNone(stager.synchronize_staging())

    def test_synchronize_with_cache_enabled(self):
        """synchronize_staging works with cache_staged_state_dict=True."""
        stager = BlockingAsyncStager(cache_staged_state_dict=True)
        state_dict = {"w": torch.randn(3, 4, device=self.device_name)}
        stager.stage(state_dict)
        self.assertIsNone(stager.synchronize_staging())

    def test_synchronize_with_type_check_enabled(self):
        """synchronize_staging works with type_check=True."""
        stager = BlockingAsyncStager(type_check=True)
        self.assertIsNone(stager.synchronize_staging())

    def test_synchronize_before_stage(self):
        """synchronize_staging before any stage call is safe."""
        stager = BlockingAsyncStager()
        self.assertIsNone(stager.synchronize_staging())


if __name__ == "__main__":
    run_tests()
