"""
torch.distributed._composable.fsdp.MixedPrecisionPolicy 单元测试

覆盖维度表:
| 维度         | 覆盖内容                                                    |
|-------------|------------------------------------------------------------|
| 默认值       | param_dtype=None, reduce_dtype=None, output_dtype=None,   |
|              | cast_forward_inputs=True                                   |
| dtype 参数   | float16, float32, bfloat16, None（含混合 dtype 组合）        |
| 布尔参数     | cast_forward_inputs=True/False                              |
| frozen 行为  | 属性赋值抛 FrozenInstanceError, __delattr__ 抛 FrozenInstanceError |
| 等价性       | 相同参数实例等价, 不同参数实例不等价                          |
| 类型检查     | 实例类型为 dataclass, 各属性类型正确                         |
| 导入路径     | torch.distributed._composable.fsdp.MixedPrecisionPolicy    |
"""

import dataclasses
import unittest

import torch
from torch_npu.testing.testcase import TestCase, run_tests


class TestComposableFsdpMixedPrecisionPolicy(TestCase):
    """Test torch.distributed._composable.fsdp.MixedPrecisionPolicy."""

    def setUp(self):
        # Verify NPU backend is available
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, "npu")

    # ------------------------------------------------------------------
    # Default values
    # ------------------------------------------------------------------

    def test_default_param_dtype_is_none(self):
        """param_dtype defaults to None."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy()
        self.assertIsNone(policy.param_dtype)

    def test_default_reduce_dtype_is_none(self):
        """reduce_dtype defaults to None."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy()
        self.assertIsNone(policy.reduce_dtype)

    def test_default_output_dtype_is_none(self):
        """output_dtype defaults to None."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy()
        self.assertIsNone(policy.output_dtype)

    def test_default_cast_forward_inputs_is_true(self):
        """cast_forward_inputs defaults to True."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy()
        self.assertTrue(policy.cast_forward_inputs)

    # ------------------------------------------------------------------
    # Explicit dtype arguments
    # ------------------------------------------------------------------

    def test_param_dtype_float16(self):
        """param_dtype can be set to torch.float16."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy(param_dtype=torch.float16)
        self.assertEqual(policy.param_dtype, torch.float16)

    def test_param_dtype_float32(self):
        """param_dtype can be set to torch.float32."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy(param_dtype=torch.float32)
        self.assertEqual(policy.param_dtype, torch.float32)

    def test_param_dtype_bfloat16(self):
        """param_dtype can be set to torch.bfloat16."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
        self.assertEqual(policy.param_dtype, torch.bfloat16)

    def test_param_dtype_explicit_none(self):
        """param_dtype can be explicitly set to None."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy(param_dtype=None)
        self.assertIsNone(policy.param_dtype)

    def test_reduce_dtype_float16(self):
        """reduce_dtype can be set to torch.float16."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy(reduce_dtype=torch.float16)
        self.assertEqual(policy.reduce_dtype, torch.float16)

    def test_reduce_dtype_float32(self):
        """reduce_dtype can be set to torch.float32."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy(reduce_dtype=torch.float32)
        self.assertEqual(policy.reduce_dtype, torch.float32)

    def test_reduce_dtype_bfloat16(self):
        """reduce_dtype can be set to torch.bfloat16."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy(reduce_dtype=torch.bfloat16)
        self.assertEqual(policy.reduce_dtype, torch.bfloat16)

    def test_reduce_dtype_explicit_none(self):
        """reduce_dtype can be explicitly set to None."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy(reduce_dtype=None)
        self.assertIsNone(policy.reduce_dtype)

    def test_output_dtype_float16(self):
        """output_dtype can be set to torch.float16."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy(output_dtype=torch.float16)
        self.assertEqual(policy.output_dtype, torch.float16)

    def test_output_dtype_float32(self):
        """output_dtype can be set to torch.float32."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy(output_dtype=torch.float32)
        self.assertEqual(policy.output_dtype, torch.float32)

    def test_output_dtype_bfloat16(self):
        """output_dtype can be set to torch.bfloat16."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy(output_dtype=torch.bfloat16)
        self.assertEqual(policy.output_dtype, torch.bfloat16)

    def test_output_dtype_explicit_none(self):
        """output_dtype can be explicitly set to None."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy(output_dtype=None)
        self.assertIsNone(policy.output_dtype)

    # ------------------------------------------------------------------
    # Combined dtype configurations
    # ------------------------------------------------------------------

    def test_all_float16(self):
        """All dtype fields set to float16."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            output_dtype=torch.float16,
        )
        self.assertEqual(policy.param_dtype, torch.float16)
        self.assertEqual(policy.reduce_dtype, torch.float16)
        self.assertEqual(policy.output_dtype, torch.float16)

    def test_mixed_dtypes(self):
        """Different dtype fields set to different types."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy(
            param_dtype=torch.float16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
        )
        self.assertEqual(policy.param_dtype, torch.float16)
        self.assertEqual(policy.reduce_dtype, torch.float32)
        self.assertEqual(policy.output_dtype, torch.bfloat16)

    def test_partial_dtype_specification(self):
        """Only param_dtype and reduce_dtype set, output_dtype left default."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        self.assertEqual(policy.param_dtype, torch.bfloat16)
        self.assertEqual(policy.reduce_dtype, torch.float32)
        self.assertIsNone(policy.output_dtype)

    # ------------------------------------------------------------------
    # cast_forward_inputs True / False
    # ------------------------------------------------------------------

    def test_cast_forward_inputs_true(self):
        """cast_forward_inputs can be set to True."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy(cast_forward_inputs=True)
        self.assertTrue(policy.cast_forward_inputs)

    def test_cast_forward_inputs_false(self):
        """cast_forward_inputs can be set to False."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy(cast_forward_inputs=False)
        self.assertFalse(policy.cast_forward_inputs)

    def test_cast_forward_inputs_with_dtypes(self):
        """cast_forward_inputs=False combined with dtype settings."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            output_dtype=torch.float32,
            cast_forward_inputs=False,
        )
        self.assertEqual(policy.param_dtype, torch.float16)
        self.assertEqual(policy.reduce_dtype, torch.float16)
        self.assertEqual(policy.output_dtype, torch.float32)
        self.assertFalse(policy.cast_forward_inputs)

    # ------------------------------------------------------------------
    # Frozen dataclass behavior
    # ------------------------------------------------------------------

    def test_frozen_setattr_raises(self):
        """Frozen dataclass raises FrozenInstanceError on attribute assignment."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy(param_dtype=torch.float16)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            policy.param_dtype = torch.float32

    def test_frozen_delattr_raises(self):
        """Frozen dataclass raises FrozenInstanceError on attribute deletion."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy(param_dtype=torch.float16)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            del policy.param_dtype

    def test_frozen_setattr_cast_forward_inputs_raises(self):
        """Frozen dataclass raises FrozenInstanceError when mutating cast_forward_inputs."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy(cast_forward_inputs=True)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            policy.cast_forward_inputs = False

    def test_frozen_setattr_reduce_dtype_raises(self):
        """Frozen dataclass raises FrozenInstanceError when mutating reduce_dtype."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy(reduce_dtype=torch.float16)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            policy.reduce_dtype = torch.float32

    # ------------------------------------------------------------------
    # Equality checks
    # ------------------------------------------------------------------

    def test_equal_default_instances(self):
        """Two default instances are equal."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy_a = MixedPrecisionPolicy()
        policy_b = MixedPrecisionPolicy()
        self.assertEqual(policy_a, policy_b)

    def test_equal_explicit_instances(self):
        """Two instances with identical explicit parameters are equal."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy_a = MixedPrecisionPolicy(
            param_dtype=torch.float16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
            cast_forward_inputs=False,
        )
        policy_b = MixedPrecisionPolicy(
            param_dtype=torch.float16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
            cast_forward_inputs=False,
        )
        self.assertEqual(policy_a, policy_b)

    def test_not_equal_different_param_dtype(self):
        """Instances with different param_dtype are not equal."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy_a = MixedPrecisionPolicy(param_dtype=torch.float16)
        policy_b = MixedPrecisionPolicy(param_dtype=torch.float32)
        self.assertNotEqual(policy_a, policy_b)

    def test_not_equal_different_reduce_dtype(self):
        """Instances with different reduce_dtype are not equal."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy_a = MixedPrecisionPolicy(reduce_dtype=torch.float16)
        policy_b = MixedPrecisionPolicy(reduce_dtype=torch.bfloat16)
        self.assertNotEqual(policy_a, policy_b)

    def test_not_equal_different_output_dtype(self):
        """Instances with different output_dtype are not equal."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy_a = MixedPrecisionPolicy(output_dtype=torch.float16)
        policy_b = MixedPrecisionPolicy(output_dtype=torch.float32)
        self.assertNotEqual(policy_a, policy_b)

    def test_not_equal_different_cast_forward_inputs(self):
        """Instances with different cast_forward_inputs are not equal."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy_a = MixedPrecisionPolicy(cast_forward_inputs=True)
        policy_b = MixedPrecisionPolicy(cast_forward_inputs=False)
        self.assertNotEqual(policy_a, policy_b)

    # ------------------------------------------------------------------
    # Type checks
    # ------------------------------------------------------------------

    def test_is_dataclass_instance(self):
        """Instance is a dataclass."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy()
        self.assertTrue(dataclasses.is_dataclass(policy))

    def test_dataclass_fields_count(self):
        """Dataclass has exactly 4 fields."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        fields = dataclasses.fields(MixedPrecisionPolicy)
        self.assertEqual(len(fields), 4)

    def test_dataclass_field_names(self):
        """Dataclass fields have the expected names."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        fields = dataclasses.fields(MixedPrecisionPolicy)
        field_names = {f.name for f in fields}
        expected_names = {"param_dtype", "reduce_dtype", "output_dtype", "cast_forward_inputs"}
        self.assertEqual(field_names, expected_names)

    def test_frozen_attribute_is_true(self):
        """Dataclass is frozen."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        self.assertTrue(dataclasses.is_dataclass(MixedPrecisionPolicy))
        # Verify frozen via __dataclass_params__
        params = MixedPrecisionPolicy.__dataclass_params__
        self.assertTrue(params.frozen)

    def test_str_representation(self):
        """String representation contains field names and values."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy(param_dtype=torch.float16)
        s = str(policy)
        self.assertIn("param_dtype", s)
        self.assertIn("float16", s)

    def test_repr_representation(self):
        """Repr is consistent with str for dataclasses."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy(param_dtype=torch.float16)
        self.assertEqual(repr(policy), str(policy))


if __name__ == "__main__":
    run_tests()
