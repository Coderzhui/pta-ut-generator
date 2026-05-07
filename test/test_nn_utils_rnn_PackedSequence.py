# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.nn.utils.rnn.PackedSequence 接口功能正确性
API 名称：torch.nn.utils.rnn.PackedSequence
API 签名：torch.nn.utils.rnn.PackedSequence(data, batch_sizes, sorted_indices=None, unsorted_indices=None)

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础创建         | pack_padded_sequence 返回正确类型   | 已覆盖                  |
| data 属性        | data 为 Tensor                      | 已覆盖                  |
| batch_sizes 属性 | batch_sizes 为 Tensor               | 已覆盖                  |
| NPU 上执行       | NPU tensor 输入                     | 已覆盖                  |
| pad_packed       | pad_packed_sequence 还原            | 已覆盖                  |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性，不做精度和数值正确性校验。
"""
import torch
import torch_npu  # noqa: F401
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    import unittest
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


class TestNnUtilsRnnPackedSequence(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")
        self.device = torch.device(self.device_name)

    def test_npu_pack_padded_sequence(self):
        """Verify pack_padded_sequence returns PackedSequence with Tensor attributes."""
        sequences = torch.randn(5, 2, 4, device=self.device)
        lengths = torch.tensor([5, 3])
        packed = pack_padded_sequence(sequences, lengths, enforce_sorted=True)
        self.assertIsInstance(packed.data, torch.Tensor)
        self.assertIsInstance(packed.batch_sizes, torch.Tensor)

    def test_npu_data_attribute_device(self):
        """Verify packed data resides on NPU device."""
        sequences = torch.randn(5, 2, 4, device=self.device)
        lengths = torch.tensor([5, 3])
        packed = pack_padded_sequence(sequences, lengths, enforce_sorted=True)
        self.assertEqual(packed.data.device.type, self.device_name)

    def test_npu_pad_packed_roundtrip(self):
        """Verify pack -> pad roundtrip preserves original shape."""
        sequences = torch.randn(5, 2, 4, device=self.device)
        lengths = torch.tensor([5, 3])
        packed = pack_padded_sequence(sequences, lengths, enforce_sorted=True)
        unpacked, out_lengths = pad_packed_sequence(packed)
        self.assertEqual(unpacked.shape, torch.Size([5, 2, 4]))

    def test_cpu_baseline(self):
        """CPU baseline: verify pack_padded_sequence on CPU."""
        sequences = torch.randn(5, 2, 4)
        lengths = torch.tensor([5, 3])
        packed = pack_padded_sequence(sequences, lengths, enforce_sorted=True)
        self.assertIsInstance(packed.data, torch.Tensor)


if __name__ == "__main__":
    run_tests()
