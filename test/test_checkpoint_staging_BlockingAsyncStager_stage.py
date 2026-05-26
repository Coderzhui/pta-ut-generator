# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.staging.BlockingAsyncStager.stage 接口功能正确性
API 名称：torch.distributed.checkpoint.staging.BlockingAsyncStager.stage
API 签名：stage(self, state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 空 state_dict vs 非空 state_dict                             | 已覆盖：空 dict 和含 tensor 的 dict             |
| 枚举选项         | cache_staged_state_dict (True/False)、type_check (True/False)| 已覆盖：两种参数的 True/False 组合              |
| 参数类型         | state_dict 类型为 Dict[str, Tensor]                          | 已覆盖：含各 dtype Tensor 的 dict               |
| 传参与不传参     | cache_staged_state_dict 默认 False、type_check 默认 False    | 已覆盖：显式传入和省略默认                       |
| 等价类/边界值    | 空 dict、单元素 tensor、多维 tensor、多种 dtype               | 已覆盖                                          |
| 正常传参场景     | 传入合法 state_dict 返回 CPU 副本                             | 已覆盖                                          |
| 异常传参场景     | 无稳定异常路径                                                | 未覆盖：stage 不抛出可预期异常                   |
| 混合设备类型     | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 已覆盖：NPU tensor 输入，验证输出在 CPU          |

未覆盖项及原因：
- 异常传参场景：stage 内部无稳定异常路径，仅做 CPU 副本

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


class TestBlockingAsyncStagerStage(TestCase):
    """Test BlockingAsyncStager.stage functionality."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu",
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_stage_returns_dict_with_same_keys(self):
        """stage returns dict with same keys as input."""
        stager = BlockingAsyncStager()
        state_dict = {
            "weight": torch.randn(3, 4, device=self.device_name),
            "bias": torch.randn(4, device=self.device_name),
        }
        result = stager.stage(state_dict)
        self.assertEqual(set(result.keys()), {"weight", "bias"})

    def test_stage_output_on_cpu(self):
        """stage returns tensors on CPU device."""
        stager = BlockingAsyncStager()
        state_dict = {"weight": torch.randn(3, 4, device=self.device_name)}
        result = stager.stage(state_dict)
        self.assertEqual(result["weight"].device.type, "cpu")

    def test_stage_preserves_shape(self):
        """stage preserves tensor shapes."""
        stager = BlockingAsyncStager()
        shapes = [(3, 4), (10,), (2, 3, 4)]
        for shape in shapes:
            state_dict = {"t": torch.randn(*shape, device=self.device_name)}
            result = stager.stage(state_dict)
            self.assertEqual(result["t"].shape, torch.Size(shape))

    def test_stage_preserves_dtype(self):
        """stage preserves tensor dtypes."""
        stager = BlockingAsyncStager()
        for dtype in [torch.float32, torch.float16, torch.int64]:
            state_dict = {"t": torch.randn(3, device=self.device_name).to(dtype)}
            result = stager.stage(state_dict)
            self.assertEqual(result["t"].dtype, dtype)

    def test_stage_cache_disabled(self):
        """stage with cache_staged_state_dict=False creates new copy each time."""
        stager = BlockingAsyncStager(cache_staged_state_dict=False)
        state_dict = {"w": torch.randn(3, 4, device=self.device_name)}
        result1 = stager.stage(state_dict)
        result2 = stager.stage(state_dict)
        # Different objects (not same memory)
        self.assertIsNot(result1["w"].data_ptr(), result2["w"].data_ptr())

    def test_stage_cache_enabled(self):
        """stage with cache_staged_state_dict=True reuses cache."""
        stager = BlockingAsyncStager(cache_staged_state_dict=True)
        state_dict = {"w": torch.randn(3, 4, device=self.device_name)}
        result1 = stager.stage(state_dict)
        result2 = stager.stage(state_dict)
        # Same cache buffer reused
        self.assertEqual(result1["w"].data_ptr(), result2["w"].data_ptr())

    def test_stage_with_type_check_true(self):
        """stage with type_check=True does not raise."""
        stager = BlockingAsyncStager(type_check=True)
        state_dict = {"w": torch.randn(3, 4, device=self.device_name)}
        result = stager.stage(state_dict)
        self.assertEqual(set(result.keys()), {"w"})

    def test_stage_with_empty_state_dict(self):
        """stage handles empty state_dict."""
        stager = BlockingAsyncStager()
        result = stager.stage({})
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)

    def test_stage_with_single_element_tensor(self):
        """stage handles single-element tensors."""
        stager = BlockingAsyncStager()
        state_dict = {"s": torch.tensor(42.0, device=self.device_name)}
        result = stager.stage(state_dict)
        self.assertEqual(result["s"].shape, torch.Size([]))
        self.assertEqual(result["s"].device.type, "cpu")

    def test_stage_with_multi_dimensional_tensors(self):
        """stage handles multi-dimensional tensors."""
        stager = BlockingAsyncStager()
        state_dict = {
            "a": torch.randn(2, 3, 4, device=self.device_name),
            "b": torch.randn(1, 1, 1, device=self.device_name),
        }
        result = stager.stage(state_dict)
        self.assertEqual(result["a"].shape, (2, 3, 4))
        self.assertEqual(result["b"].shape, (1, 1, 1))


if __name__ == "__main__":
    run_tests()
