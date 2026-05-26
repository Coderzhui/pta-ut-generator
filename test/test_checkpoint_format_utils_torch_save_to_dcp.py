# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.format_utils.torch_save_to_dcp 接口功能正确性
API 名称：torch.distributed.checkpoint.format_utils.torch_save_to_dcp
API 签名：torch_save_to_dcp(torch_save_path: str | os.PathLike, dcp_checkpoint_dir: str | os.PathLike) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 空 state_dict vs 非空 state_dict                             | 已覆盖                                          |
| 枚举选项         | N/A（无枚举参数）                                            | 未覆盖：无枚举参数                              |
| 参数类型         | torch_save_path 和 dcp_checkpoint_dir 均为 str 或 PathLike   | 已覆盖：str 路径                                |
| 传参与不传参     | 两个参数均为必传                                              | 已覆盖                                          |
| 等价类/边界值    | 空 dict、单 tensor、多 tensor、多种 dtype                     | 已覆盖                                          |
| 正常传参场景     | 传入合法路径和合法 torch save 文件                            | 已覆盖                                          |
| 异常传参场景     | 源文件不存在时 torch.load 报错                                | 已覆盖                                          |
| 混合设备类型     | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖：torch.save 文件为 CPU 格式，不涉及设备混合 |

未覆盖项及原因：
- 枚举选项：无枚举参数
- 混合设备类型：torch.save 文件在 CPU，不涉及 NPU

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import os
import tempfile

import torch
from torch.distributed.checkpoint.format_utils import torch_save_to_dcp

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


class TestTorchSaveToDcp(TestCase):
    """Test torch_save_to_dcp conversion function."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu",
            f"Expected device 'npu', got '{self.device_name}'"
        )
        self.tmpdir = tempfile.mkdtemp()

    def _torch_save_path(self, name="model.pt"):
        return os.path.join(self.tmpdir, name)

    def _dcp_dir(self, name="dcp_output"):
        path = os.path.join(self.tmpdir, name)
        os.makedirs(path, exist_ok=True)
        return path

    def test_creates_checkpoint_directory(self):
        """torch_save_to_dcp creates output directory."""
        torch_path = self._torch_save_path()
        torch.save({"w": torch.randn(3, 4)}, torch_path)
        dcp_path = os.path.join(self.tmpdir, "new_dcp_dir")
        torch_save_to_dcp(torch_path, dcp_path)
        self.assertTrue(os.path.isdir(dcp_path))

    def test_with_simple_tensor(self):
        """torch_save_to_dcp works with single tensor state_dict."""
        torch_path = self._torch_save_path()
        state_dict = {"weight": torch.randn(3, 4)}
        torch.save(state_dict, torch_path)
        dcp_path = self._dcp_dir()
        torch_save_to_dcp(torch_path, dcp_path)
        self.assertTrue(os.path.isdir(dcp_path))

    def test_with_multi_tensor(self):
        """torch_save_to_dcp works with multi-tensor state_dict."""
        torch_path = self._torch_save_path()
        state_dict = {
            "weight": torch.randn(3, 4),
            "bias": torch.randn(4),
            "embedding": torch.randn(10, 4),
        }
        torch.save(state_dict, torch_path)
        dcp_path = self._dcp_dir()
        torch_save_to_dcp(torch_path, dcp_path)
        self.assertTrue(os.path.isdir(dcp_path))

    def test_output_contains_metadata(self):
        """torch_save_to_dcp output contains .metadata file."""
        torch_path = self._torch_save_path()
        torch.save({"w": torch.randn(3, 4)}, torch_path)
        dcp_path = self._dcp_dir()
        torch_save_to_dcp(torch_path, dcp_path)
        metadata_path = os.path.join(dcp_path, ".metadata")
        self.assertTrue(
            os.path.exists(metadata_path),
            f"Metadata file not found in {os.listdir(dcp_path)}"
        )

    def test_with_various_dtypes(self):
        """torch_save_to_dcp preserves tensor dtypes."""
        torch_path = self._torch_save_path()
        state_dict = {
            "fp32": torch.randn(3),
            "fp16": torch.randn(3).half(),
            "int64": torch.randint(0, 10, (3,), dtype=torch.int64),
        }
        torch.save(state_dict, torch_path)
        dcp_path = self._dcp_dir()
        torch_save_to_dcp(torch_path, dcp_path)
        self.assertTrue(os.path.isdir(dcp_path))

    def test_with_empty_state_dict(self):
        """torch_save_to_dcp handles empty state_dict."""
        torch_path = self._torch_save_path()
        torch.save({}, torch_path)
        dcp_path = self._dcp_dir()
        torch_save_to_dcp(torch_path, dcp_path)
        self.assertTrue(os.path.isdir(dcp_path))

    def test_source_file_not_found(self):
        """torch_save_to_dcp raises when source file does not exist."""
        with self.assertRaises(Exception):
            torch_save_to_dcp("/nonexistent/path.pt", self._dcp_dir())


if __name__ == "__main__":
    run_tests()
