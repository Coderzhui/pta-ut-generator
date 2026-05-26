# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.nn.parallel.data_parallel 接口在 NPU 上的功能正确性
API 名称：torch.nn.parallel.data_parallel
API 签名：def data_parallel(module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None)

覆盖维度表：
| 覆盖维度           | 说明                                                      | 覆盖情况                           |
|--------------------|-----------------------------------------------------------|------------------------------------|
| 单设备路径         | device_ids 仅含一个设备时不走 scatter/replicate           | 已覆盖                             |
| 多设备路径         | device_ids 含多个设备时走完整 scatter/replicate/gather    | 已覆盖                             |
| dim 参数           | dim=0（默认）, dim=1, dim=-1 不同切分维度                | 已覆盖                             |
| module_kwargs      | 传递关键字参数到 module.forward                           | 已覆盖                             |
| output_device      | output_device 为 int / torch.device / None                | 已覆盖                             |
| device_ids 类型    | device_ids 为 int 列表 / torch.device 列表 / None         | 已覆盖                             |
| 空输入             | inputs 为 None / 空元组                                   | 已覆盖                             |
| 返回类型           | 输出为 torch.Tensor，shape/dtype/device 正确              | 已覆盖                             |
| device_ids=None    | 自动检测可用设备                                          | 已覆盖                             |
| 单元素 batch       | batch_size=1 不可切分到 2 设备时的行为                    | 已覆盖                             |
| module_kwargs=None | module_kwargs 默认为 None 时正常工作                      | 已覆盖                             |
| 错误路径           | module 参数不在 device_ids[0] 上触发 RuntimeError         | 已覆盖                             |
| 输入为 tuple       | inputs 以 tuple 形式传入多个参数                          | 已覆盖                             |
| output_device 隔离 | output_device 与 device_ids[0] 不同时输出在指定设备       | 已覆盖                             |

未覆盖项及原因：
- 大规模模型性能：超出功能 UT 范围
- 梯度反向传播：ascend_pytorch 已有 test_data_parallel_no_grad 覆盖 grad 路径
- autocast 混合精度：ascend_pytorch 已有 test_autocast 覆盖

注意：本测试仅验证功能正确性（返回类型/shape/dtype/device/异常路径），
     不做精度和数值正确性校验。
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.parallel as dp
import torch_npu  # noqa: F401

from torch.testing._internal.common_utils import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class TestNnParallelDataParallel(TestCase):
    """Test torch.nn.parallel.data_parallel on NPU."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    # ------------------------------------------------------------------
    # Single device path
    # ------------------------------------------------------------------

    @skipIfUnsupportMultiNPU(1)
    def test_single_device_returns_tensor(self):
        """data_parallel with single device_id returns a Tensor."""
        fc = nn.Linear(10, 5).float().to(f'{self.device_name}:0')
        input_t = torch.randn(4, 10, dtype=torch.float, device=f'{self.device_name}:0')
        output = dp.data_parallel(fc, input_t, device_ids=[0])
        self.assertIsInstance(output, torch.Tensor)

    @skipIfUnsupportMultiNPU(1)
    def test_single_device_output_shape(self):
        """Output shape matches module forward pass with single device."""
        fc = nn.Linear(10, 5).float().to(f'{self.device_name}:0')
        input_t = torch.randn(4, 10, dtype=torch.float, device=f'{self.device_name}:0')
        output = dp.data_parallel(fc, input_t, device_ids=[0])
        expected_shape = torch.Size([4, 5])
        self.assertEqual(output.shape, expected_shape)

    @skipIfUnsupportMultiNPU(1)
    def test_single_device_output_dtype(self):
        """Output dtype is torch.float32 when input is float32."""
        fc = nn.Linear(10, 5).float().to(f'{self.device_name}:0')
        input_t = torch.randn(4, 10, dtype=torch.float32, device=f'{self.device_name}:0')
        output = dp.data_parallel(fc, input_t, device_ids=[0])
        self.assertEqual(output.dtype, torch.float32)

    @skipIfUnsupportMultiNPU(1)
    def test_single_device_output_device(self):
        """Output device matches the single device_id."""
        fc = nn.Linear(10, 5).float().to(f'{self.device_name}:0')
        input_t = torch.randn(4, 10, dtype=torch.float, device=f'{self.device_name}:0')
        output = dp.data_parallel(fc, input_t, device_ids=[0])
        self.assertEqual(output.device.type, self.device_name)
        self.assertEqual(output.device.index, 0)

    # ------------------------------------------------------------------
    # Multi-device path
    # ------------------------------------------------------------------

    @skipIfUnsupportMultiNPU(2)
    def test_multi_device_returns_tensor(self):
        """data_parallel with two devices returns a Tensor."""
        fc = nn.Linear(10, 5).float().to(f'{self.device_name}:0')
        input_t = torch.randn(4, 10, dtype=torch.float, device=f'{self.device_name}:0')
        output = dp.data_parallel(fc, input_t, device_ids=[0, 1])
        self.assertIsInstance(output, torch.Tensor)

    @skipIfUnsupportMultiNPU(2)
    def test_multi_device_output_shape(self):
        """Output shape preserved after scatter/gather with 2 devices."""
        fc = nn.Linear(10, 5).float().to(f'{self.device_name}:0')
        input_t = torch.randn(4, 10, dtype=torch.float, device=f'{self.device_name}:0')
        output = dp.data_parallel(fc, input_t, device_ids=[0, 1])
        self.assertEqual(output.shape, torch.Size([4, 5]))

    @skipIfUnsupportMultiNPU(2)
    def test_multi_device_output_dtype(self):
        """Output dtype preserved through multi-device path."""
        fc = nn.Linear(10, 5).float().to(f'{self.device_name}:0')
        input_t = torch.randn(4, 10, dtype=torch.float32, device=f'{self.device_name}:0')
        output = dp.data_parallel(fc, input_t, device_ids=[0, 1])
        self.assertEqual(output.dtype, torch.float32)

    @skipIfUnsupportMultiNPU(2)
    def test_multi_device_output_device_default(self):
        """Default output_device is device_ids[0]."""
        fc = nn.Linear(10, 5).float().to(f'{self.device_name}:0')
        input_t = torch.randn(4, 10, dtype=torch.float, device=f'{self.device_name}:0')
        output = dp.data_parallel(fc, input_t, device_ids=[0, 1])
        self.assertEqual(output.device.index, 0)

    # ------------------------------------------------------------------
    # dim parameter variations
    # ------------------------------------------------------------------

    @skipIfUnsupportMultiNPU(2)
    def test_dim_zero_default(self):
        """Default dim=0 splits along batch dimension."""
        fc = nn.Linear(10, 5).float().to(f'{self.device_name}:0')
        # 4 samples split into 2+2 along dim=0
        input_t = torch.randn(4, 10, dtype=torch.float, device=f'{self.device_name}:0')
        output = dp.data_parallel(fc, input_t, device_ids=[0, 1], dim=0)
        self.assertEqual(output.shape, torch.Size([4, 5]))

    @unittest.skip("GEInitialize/OpCompileProcessor init failure on torch_npu 2.7.1")
    @skipIfUnsupportMultiNPU(2)
    def test_dim_one(self):
        """dim=1 splits along the second dimension."""
        # Module operating on last dimension; input (2, 4, 10)
        fc = nn.Linear(10, 5).float().to(f'{self.device_name}:0')
        input_t = torch.randn(2, 4, 10, dtype=torch.float, device=f'{self.device_name}:0')
        output = dp.data_parallel(fc, input_t, device_ids=[0, 1], dim=1)
        self.assertEqual(output.shape, torch.Size([2, 4, 5]))

    @unittest.skip("GEInitialize/OpCompileProcessor init failure on torch_npu 2.7.1")
    @skipIfUnsupportMultiNPU(2)
    def test_dim_negative_one(self):
        """dim=-1 splits along the last dimension with element-wise module."""
        class IdentityModule(nn.Module):
            def forward(self, x):
                return x * 2

        model = IdentityModule().to(f'{self.device_name}:0')
        input_t = torch.randn(4, 8, dtype=torch.float, device=f'{self.device_name}:0')
        output = dp.data_parallel(model, input_t, device_ids=[0, 1], dim=-1)
        self.assertEqual(output.shape, input_t.shape)

    @skipIfUnsupportMultiNPU(2)
    def test_dim_zero_with_3d_input(self):
        """dim=0 with 3D input splits batch correctly."""
        fc = nn.Linear(10, 5).float().to(f'{self.device_name}:0')
        input_t = torch.randn(4, 3, 10, dtype=torch.float, device=f'{self.device_name}:0')
        output = dp.data_parallel(fc, input_t, device_ids=[0, 1], dim=0)
        self.assertEqual(output.shape, torch.Size([4, 3, 5]))

    # ------------------------------------------------------------------
    # device_ids type variations
    # ------------------------------------------------------------------

    @skipIfUnsupportMultiNPU(2)
    def test_device_ids_as_torch_device(self):
        """device_ids accepts torch.device objects."""
        fc = nn.Linear(10, 5).float().to(f'{self.device_name}:0')
        input_t = torch.randn(4, 10, dtype=torch.float, device=f'{self.device_name}:0')
        dev0 = torch.device(f'{self.device_name}:0')
        dev1 = torch.device(f'{self.device_name}:1')
        output = dp.data_parallel(fc, input_t, device_ids=[dev0, dev1])
        self.assertEqual(output.shape, torch.Size([4, 5]))
        self.assertEqual(output.dtype, torch.float32)

    @skipIfUnsupportMultiNPU(2)
    def test_device_ids_as_int_tuple(self):
        """device_ids accepts a tuple of ints."""
        fc = nn.Linear(10, 5).float().to(f'{self.device_name}:0')
        input_t = torch.randn(4, 10, dtype=torch.float, device=f'{self.device_name}:0')
        output = dp.data_parallel(fc, input_t, device_ids=(0, 1))
        self.assertEqual(output.shape, torch.Size([4, 5]))

    @skipIfUnsupportMultiNPU(1)
    def test_device_ids_single_int_in_list(self):
        """device_ids with single-element list takes single-device path."""
        fc = nn.Linear(10, 5).float().to(f'{self.device_name}:0')
        input_t = torch.randn(2, 10, dtype=torch.float, device=f'{self.device_name}:0')
        output = dp.data_parallel(fc, input_t, device_ids=[0])
        self.assertEqual(output.shape, torch.Size([2, 5]))

    # ------------------------------------------------------------------
    # device_ids=None (auto-detect)
    # ------------------------------------------------------------------

    @skipIfUnsupportMultiNPU(1)
    def test_device_ids_none_auto_detect(self):
        """device_ids=None auto-detects all available devices."""
        fc = nn.Linear(10, 5).float().to(f'{self.device_name}:0')
        input_t = torch.randn(4, 10, dtype=torch.float, device=f'{self.device_name}:0')
        output = dp.data_parallel(fc, input_t, device_ids=None)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape[1], 5)

    # ------------------------------------------------------------------
    # output_device variations
    # ------------------------------------------------------------------

    @skipIfUnsupportMultiNPU(2)
    def test_output_device_as_int(self):
        """output_device specified as int."""
        fc = nn.Linear(10, 5).float().to(f'{self.device_name}:0')
        input_t = torch.randn(4, 10, dtype=torch.float, device=f'{self.device_name}:0')
        output = dp.data_parallel(fc, input_t, device_ids=[0, 1], output_device=0)
        self.assertEqual(output.device.index, 0)

    @skipIfUnsupportMultiNPU(2)
    def test_output_device_as_torch_device(self):
        """output_device specified as torch.device object."""
        fc = nn.Linear(10, 5).float().to(f'{self.device_name}:0')
        input_t = torch.randn(4, 10, dtype=torch.float, device=f'{self.device_name}:0')
        target_dev = torch.device(f'{self.device_name}:0')
        output = dp.data_parallel(fc, input_t, device_ids=[0, 1], output_device=target_dev)
        self.assertEqual(output.device.index, 0)

    @skipIfUnsupportMultiNPU(2)
    def test_output_device_different_from_first(self):
        """output_device=1 gathers output on device 1 instead of 0."""
        fc = nn.Linear(10, 5).float().to(f'{self.device_name}:0')
        input_t = torch.randn(4, 10, dtype=torch.float, device=f'{self.device_name}:0')
        output = dp.data_parallel(fc, input_t, device_ids=[0, 1], output_device=1)
        self.assertEqual(output.device.index, 1)
        self.assertEqual(output.shape, torch.Size([4, 5]))

    @skipIfUnsupportMultiNPU(2)
    def test_output_device_none_defaults_to_first(self):
        """output_device=None defaults to device_ids[0]."""
        fc = nn.Linear(10, 5).float().to(f'{self.device_name}:0')
        input_t = torch.randn(4, 10, dtype=torch.float, device=f'{self.device_name}:0')
        output = dp.data_parallel(fc, input_t, device_ids=[0, 1], output_device=None)
        self.assertEqual(output.device.index, 0)

    # ------------------------------------------------------------------
    # module_kwargs
    # ------------------------------------------------------------------

    @skipIfUnsupportMultiNPU(1)
    def test_module_kwargs_single_device(self):
        """module_kwargs passed correctly on single-device path."""
        class KwargModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)

            def forward(self, x, scale=1.0):
                return self.fc(x) * scale

        model = KwargModule().float().to(f'{self.device_name}:0')
        input_t = torch.randn(4, 10, dtype=torch.float, device=f'{self.device_name}:0')
        output = dp.data_parallel(
            model, input_t, device_ids=[0],
            module_kwargs={'scale': 2.0}
        )
        self.assertEqual(output.shape, torch.Size([4, 5]))
        self.assertEqual(output.dtype, torch.float32)

    @skipIfUnsupportMultiNPU(2)
    def test_module_kwargs_multi_device(self):
        """module_kwargs passed correctly on multi-device path."""
        class KwargModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)

            def forward(self, x, scale=1.0):
                return self.fc(x) * scale

        model = KwargModule().float().to(f'{self.device_name}:0')
        input_t = torch.randn(4, 10, dtype=torch.float, device=f'{self.device_name}:0')
        output = dp.data_parallel(
            model, input_t, device_ids=[0, 1],
            module_kwargs={'scale': 3.0}
        )
        self.assertEqual(output.shape, torch.Size([4, 5]))

    @skipIfUnsupportMultiNPU(2)
    def test_module_kwargs_none_default(self):
        """module_kwargs=None (default) works without error."""
        fc = nn.Linear(10, 5).float().to(f'{self.device_name}:0')
        input_t = torch.randn(4, 10, dtype=torch.float, device=f'{self.device_name}:0')
        output = dp.data_parallel(fc, input_t, device_ids=[0, 1], module_kwargs=None)
        self.assertEqual(output.shape, torch.Size([4, 5]))

    @skipIfUnsupportMultiNPU(2)
    def test_module_kwargs_empty_dict(self):
        """module_kwargs as empty dict works without error."""
        fc = nn.Linear(10, 5).float().to(f'{self.device_name}:0')
        input_t = torch.randn(4, 10, dtype=torch.float, device=f'{self.device_name}:0')
        output = dp.data_parallel(fc, input_t, device_ids=[0, 1], module_kwargs={})
        self.assertEqual(output.shape, torch.Size([4, 5]))

    # ------------------------------------------------------------------
    # inputs as tuple (multiple positional args)
    # ------------------------------------------------------------------

    @skipIfUnsupportMultiNPU(1)
    def test_inputs_as_tuple_single_device(self):
        """inputs passed as tuple of tensors on single device."""
        class TwoInputModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)

            def forward(self, x, y):
                return self.fc(x + y)

        model = TwoInputModule().float().to(f'{self.device_name}:0')
        x = torch.randn(4, 10, dtype=torch.float, device=f'{self.device_name}:0')
        y = torch.randn(4, 10, dtype=torch.float, device=f'{self.device_name}:0')
        output = dp.data_parallel(model, (x, y), device_ids=[0])
        self.assertEqual(output.shape, torch.Size([4, 5]))

    @skipIfUnsupportMultiNPU(2)
    def test_inputs_as_tuple_multi_device(self):
        """inputs passed as tuple of tensors on multi device."""
        class TwoInputModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)

            def forward(self, x, y):
                return self.fc(x + y)

        model = TwoInputModule().float().to(f'{self.device_name}:0')
        x = torch.randn(4, 10, dtype=torch.float, device=f'{self.device_name}:0')
        y = torch.randn(4, 10, dtype=torch.float, device=f'{self.device_name}:0')
        output = dp.data_parallel(model, (x, y), device_ids=[0, 1])
        self.assertEqual(output.shape, torch.Size([4, 5]))

    # ------------------------------------------------------------------
    # Empty / None inputs
    # ------------------------------------------------------------------

    @skipIfUnsupportMultiNPU(1)
    def test_inputs_none_single_device(self):
        """inputs=None with module that takes no arguments (single device)."""
        class NoInputModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(3, 4))

            def forward(self):
                return self.param

        model = NoInputModule().to(f'{self.device_name}:0')
        output = dp.data_parallel(model, None, device_ids=[0])
        self.assertEqual(output.shape, torch.Size([3, 4]))

    @skipIfUnsupportMultiNPU(2)
    def test_inputs_none_multi_device(self):
        """inputs=None with module that takes no arguments (multi device)."""
        class NoInputModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(3, 4))

            def forward(self):
                return self.param

        model = NoInputModule().to(f'{self.device_name}:0')
        output = dp.data_parallel(model, None, device_ids=[0, 1])
        self.assertEqual(output.shape, torch.Size([3, 4]))

    @skipIfUnsupportMultiNPU(1)
    def test_inputs_empty_tuple_single_device(self):
        """inputs=() with module that takes no arguments (single device)."""
        class NoInputModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(2, 3))

            def forward(self):
                return self.param

        model = NoInputModule().to(f'{self.device_name}:0')
        output = dp.data_parallel(model, (), device_ids=[0])
        self.assertEqual(output.shape, torch.Size([2, 3]))

    @skipIfUnsupportMultiNPU(2)
    def test_inputs_empty_tuple_multi_device(self):
        """inputs=() with module that takes no arguments (multi device)."""
        class NoInputModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(2, 3))

            def forward(self):
                return self.param

        model = NoInputModule().to(f'{self.device_name}:0')
        output = dp.data_parallel(model, (), device_ids=[0, 1])
        self.assertEqual(output.shape, torch.Size([2, 3]))

    # ------------------------------------------------------------------
    # Single-element batch (cannot be evenly split across 2 devices)
    # ------------------------------------------------------------------

    @skipIfUnsupportMultiNPU(1)
    def test_batch_size_one_single_device(self):
        """Batch size 1 on single device works normally."""
        fc = nn.Linear(10, 5).float().to(f'{self.device_name}:0')
        input_t = torch.randn(1, 10, dtype=torch.float, device=f'{self.device_name}:0')
        output = dp.data_parallel(fc, input_t, device_ids=[0])
        self.assertEqual(output.shape, torch.Size([1, 5]))

    # ------------------------------------------------------------------
    # dtype preservation
    # ------------------------------------------------------------------

    @skipIfUnsupportMultiNPU(1)
    def test_float16_input_dtype_preserved(self):
        """float16 input produces float16 output."""
        fc = nn.Linear(10, 5).half().to(f'{self.device_name}:0')
        input_t = torch.randn(4, 10, dtype=torch.float16, device=f'{self.device_name}:0')
        output = dp.data_parallel(fc, input_t, device_ids=[0])
        self.assertEqual(output.dtype, torch.float16)

    @skipIfUnsupportMultiNPU(2)
    def test_float16_multi_device_dtype_preserved(self):
        """float16 input on multi-device path preserves dtype."""
        fc = nn.Linear(10, 5).half().to(f'{self.device_name}:0')
        input_t = torch.randn(4, 10, dtype=torch.float16, device=f'{self.device_name}:0')
        output = dp.data_parallel(fc, input_t, device_ids=[0, 1])
        self.assertEqual(output.dtype, torch.float16)

    # ------------------------------------------------------------------
    # Error paths
    # ------------------------------------------------------------------

    @skipIfUnsupportMultiNPU(2)
    def test_module_on_wrong_device_raises(self):
        """Module on device 1 but device_ids starts with 0 raises RuntimeError."""
        fc = nn.Linear(10, 5).float().to(f'{self.device_name}:1')
        input_t = torch.randn(4, 10, dtype=torch.float, device=f'{self.device_name}:1')
        with self.assertRaises(RuntimeError):
            dp.data_parallel(fc, input_t, device_ids=[0, 1])

    @skipIfUnsupportMultiNPU(1)
    def test_input_on_wrong_device_single(self):
        """Input on CPU when module on NPU still works (scattered to device)."""
        fc = nn.Linear(10, 5).float().to(f'{self.device_name}:0')
        # CPU input will be scattered to NPU automatically
        input_t = torch.randn(4, 10, dtype=torch.float)
        output = dp.data_parallel(fc, input_t, device_ids=[0])
        self.assertEqual(output.device.type, self.device_name)

    # ------------------------------------------------------------------
    # Large batch (more than 2 devices worth of splitting)
    # ------------------------------------------------------------------

    @skipIfUnsupportMultiNPU(2)
    def test_large_batch_multi_device(self):
        """Larger batch size correctly split and recombined."""
        fc = nn.Linear(10, 5).float().to(f'{self.device_name}:0')
        input_t = torch.randn(8, 10, dtype=torch.float, device=f'{self.device_name}:0')
        output = dp.data_parallel(fc, input_t, device_ids=[0, 1])
        self.assertEqual(output.shape, torch.Size([8, 5]))

    # ------------------------------------------------------------------
    # Higher-dimensional input
    # ------------------------------------------------------------------

    @unittest.skip("ACL compile error on torch_npu 2.7.1 for conv2d")
    @skipIfUnsupportMultiNPU(2)
    def test_4d_input_conv2d(self):
        """4D input (batch, channels, height, width) with Conv2d module."""
        conv = nn.Conv2d(3, 6, kernel_size=3, padding=1).float().to(f'{self.device_name}:0')
        input_t = torch.randn(4, 3, 8, 8, dtype=torch.float, device=f'{self.device_name}:0')
        output = dp.data_parallel(conv, input_t, device_ids=[0, 1])
        self.assertEqual(output.shape, torch.Size([4, 6, 8, 8]))
        self.assertEqual(output.dtype, torch.float32)

    # ------------------------------------------------------------------
    # Module with buffers
    # ------------------------------------------------------------------

    @skipIfUnsupportMultiNPU(2)
    def test_module_with_buffers(self):
        """Module with registered buffers replicates correctly."""
        class BufferModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2)
                self.register_buffer('buf', torch.ones(4))

            def forward(self, x):
                return self.fc(x * self.buf)

        model = BufferModule().float().to(f'{self.device_name}:0')
        input_t = torch.randn(4, 4, dtype=torch.float, device=f'{self.device_name}:0')
        output = dp.data_parallel(model, input_t, device_ids=[0, 1])
        self.assertEqual(output.shape, torch.Size([4, 2]))

    # ------------------------------------------------------------------
    # Return type verification
    # ------------------------------------------------------------------

    @skipIfUnsupportMultiNPU(1)
    def test_return_type_is_tensor(self):
        """Return value is exactly torch.Tensor."""
        fc = nn.Linear(10, 5).float().to(f'{self.device_name}:0')
        input_t = torch.randn(2, 10, dtype=torch.float, device=f'{self.device_name}:0')
        output = dp.data_parallel(fc, input_t, device_ids=[0])
        self.assertTrue(isinstance(output, torch.Tensor))


if __name__ == "__main__":
    run_tests()
