# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.read_metadata 接口功能正确性
API 名称：torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.read_metadata
API 签名：read_metadata(self) -> Metadata

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 返回的 Metadata 中 state_dict_metadata 为空字典              | 已覆盖                                          |
| 枚举选项         | coordinator_rank 参数 (0, 非默认值)                          | 已覆盖                                          |
| 参数类型         | checkpoint_id 类型 (str, None)                               | 已覆盖                                          |
| 传参与不传参     | checkpoint_id 默认 None vs 显式传入                          | 已覆盖                                          |
| 等价类/边界值    | 不同构造参数下的返回值一致性                                  | 已覆盖                                          |
| 正常传参场景     | 调用 read_metadata 返回 Metadata                             | 已覆盖                                          |
| 异常传参场景     | 无稳定异常路径                                                | 未覆盖：方法不抛出异常                          |
| 混合设备类型     | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖：read_metadata 不涉及设备迁移            |

未覆盖项及原因：
- 异常传参场景：read_metadata 内部无异常路径
- 混合设备类型：纯元数据操作，不涉及设备

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import torch
from torch.distributed.checkpoint.format_utils import BroadcastingTorchSaveReader
from torch.distributed.checkpoint.metadata import Metadata

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


class TestBroadcastingTorchSaveReaderReadMetadata(TestCase):
    """Test BroadcastingTorchSaveReader.read_metadata."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu",
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_returns_metadata_instance(self):
        """read_metadata returns a Metadata object."""
        reader = BroadcastingTorchSaveReader()
        result = reader.read_metadata()
        self.assertIsInstance(result, Metadata)

    def test_returns_empty_state_dict_metadata(self):
        """read_metadata returns Metadata with empty state_dict_metadata."""
        reader = BroadcastingTorchSaveReader()
        result = reader.read_metadata()
        self.assertIsInstance(result.state_dict_metadata, dict)
        self.assertEqual(len(result.state_dict_metadata), 0)

    def test_with_default_constructor(self):
        """read_metadata works with default constructor (checkpoint_id=None)."""
        reader = BroadcastingTorchSaveReader()
        result = reader.read_metadata()
        self.assertIsInstance(result, Metadata)

    def test_with_checkpoint_id_set(self):
        """read_metadata works with checkpoint_id set."""
        reader = BroadcastingTorchSaveReader(checkpoint_id="/tmp/test.pt")
        result = reader.read_metadata()
        self.assertIsInstance(result, Metadata)
        self.assertEqual(len(result.state_dict_metadata), 0)

    def test_with_custom_coordinator_rank(self):
        """read_metadata works with custom coordinator_rank."""
        reader = BroadcastingTorchSaveReader(coordinator_rank=1)
        result = reader.read_metadata()
        self.assertIsInstance(result, Metadata)

    def test_return_type_is_metadata(self):
        """read_metadata return type is exactly Metadata."""
        reader = BroadcastingTorchSaveReader()
        result = reader.read_metadata()
        self.assertIs(type(result), Metadata)

    def test_multiple_calls_return_fresh_metadata(self):
        """Multiple read_metadata calls return new Metadata objects."""
        reader = BroadcastingTorchSaveReader()
        result1 = reader.read_metadata()
        result2 = reader.read_metadata()
        self.assertIsNot(result1, result2)


if __name__ == "__main__":
    run_tests()
