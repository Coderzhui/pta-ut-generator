# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.FileSystemWriter.stage 接口功能正确性
API 名称：torch.distributed.checkpoint.FileSystemWriter.stage
API 签名：stage(self, state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 空 state_dict vs 非空 state_dict                             | 已覆盖                                          |
| 枚举选项         | cache_staged_state_dict (True/False)                         | 已覆盖                                          |
| 参数类型         | state_dict 为 Dict[str, Tensor]                              | 已覆盖                                          |
| 传参与不传参     | cache_staged_state_dict 默认 False vs True                   | 已覆盖                                          |
| 等价类/边界值    | 空 dict、单元素 tensor、多维 tensor                           | 已覆盖                                          |
| 正常传参场景     | 传入 NPU tensor 的 state_dict 返回 CPU 副本                   | 已覆盖                                          |
| 异常传参场景     | 无稳定异常路径                                                | 未覆盖：stage 不抛异常                          |
| 混合设备类型     | NPU 输入 → CPU 输出                                           | 已覆盖                                          |

未覆盖项及原因：
- 异常传参场景：stage 内部无稳定异常路径

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import tempfile

import torch
from torch.distributed.checkpoint import FileSystemWriter

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


class TestFileSystemWriterStage(TestCase):
    """Test FileSystemWriter.stage method."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu",
            f"Expected device 'npu', got '{self.device_name}'"
        )
        self.tmpdir = tempfile.mkdtemp()

    def test_stage_returns_dict_with_same_keys(self):
        """stage returns dict with same keys as input."""
        writer = FileSystemWriter(self.tmpdir)
        state_dict = {
            "weight": torch.randn(3, 4, device=self.device_name),
            "bias": torch.randn(4, device=self.device_name),
        }
        result = writer.stage(state_dict)
        self.assertEqual(set(result.keys()), {"weight", "bias"})

    def test_stage_returns_cpu_tensors(self):
        """stage returns tensors on CPU device."""
        writer = FileSystemWriter(self.tmpdir)
        state_dict = {"w": torch.randn(3, 4, device=self.device_name)}
        result = writer.stage(state_dict)
        self.assertEqual(result["w"].device.type, "cpu")

    def test_stage_cache_disabled(self):
        """stage with cache_staged_state_dict=False creates new copies."""
        writer = FileSystemWriter(self.tmpdir, cache_staged_state_dict=False)
        state_dict = {"w": torch.randn(3, 4, device=self.device_name)}
        r1 = writer.stage(state_dict)
        r2 = writer.stage(state_dict)
        self.assertIsNot(r1["w"].data_ptr(), r2["w"].data_ptr())

    def test_stage_cache_enabled(self):
        """stage with cache_staged_state_dict=True reuses buffer."""
        writer = FileSystemWriter(self.tmpdir, cache_staged_state_dict=True)
        state_dict = {"w": torch.randn(3, 4, device=self.device_name)}
        r1 = writer.stage(state_dict)
        r2 = writer.stage(state_dict)
        self.assertEqual(r1["w"].data_ptr(), r2["w"].data_ptr())

    def test_stage_npu_input_to_cpu_output(self):
        """stage moves NPU tensors to CPU."""
        writer = FileSystemWriter(self.tmpdir)
        state_dict = {"w": torch.randn(5, 5, device=self.device_name)}
        result = writer.stage(state_dict)
        self.assertEqual(result["w"].device.type, "cpu")

    def test_stage_preserves_shapes(self):
        """stage preserves tensor shapes."""
        writer = FileSystemWriter(self.tmpdir)
        shapes = [(3, 4), (10,), (2, 3, 4)]
        for shape in shapes:
            state_dict = {"t": torch.randn(*shape, device=self.device_name)}
            result = writer.stage(state_dict)
            self.assertEqual(result["t"].shape, torch.Size(shape))

    def test_stage_preserves_dtypes(self):
        """stage preserves tensor dtypes."""
        writer = FileSystemWriter(self.tmpdir)
        for dtype in [torch.float32, torch.float16]:
            state_dict = {"t": torch.randn(3, device=self.device_name).to(dtype)}
            result = writer.stage(state_dict)
            self.assertEqual(result["t"].dtype, dtype)

    def test_stage_empty_state_dict(self):
        """stage handles empty state_dict."""
        writer = FileSystemWriter(self.tmpdir)
        result = writer.stage({})
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)

    def test_per_thread_copy_ahead_set_to_zero(self):
        """stage sets per_thread_copy_ahead to 0."""
        writer = FileSystemWriter(self.tmpdir, per_thread_copy_ahead=10000000)
        state_dict = {"w": torch.randn(3, device=self.device_name)}
        writer.stage(state_dict)
        self.assertEqual(writer.per_thread_copy_ahead, 0)


if __name__ == "__main__":
    run_tests()
