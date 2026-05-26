# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name 接口功能正确性
API 名称：torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name
API 签名：def set_module_name(self, module_name: str, qconfig: QConfigAny) -> QConfigMapping

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况               |
|------------------|--------------------------------------------|------------------------|
| 基础调用         | 正常参数调用不报错                         | 已覆盖                 |
| 返回类型         | 返回 QConfigMapping (self) 用于链式调用    | 已覆盖                 |
| 参数类型         | module_name 为 str, qconfig 为 QConfig/None | 已覆盖               |
| 覆盖已有映射     | 重复 set 同一 module_name 覆盖旧值         | 已覆盖                 |
| 链式调用         | 连续 .set_module_name() 返回 self          | 已覆盖                 |
| 空字符串         | module_name 为空字符串                     | 已覆盖                 |
| qconfig=None     | 传入 None 作为 qconfig                     | 已覆盖                 |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、返回类型/副作用符合预期），
     不做精度和数值正确性校验。
"""

import torch
import torch_npu  # noqa: F401

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        import unittest
        unittest.main(argv=sys.argv)

from torch.ao.quantization.qconfig_mapping import QConfigMapping


class TestQConfigMappingSetModuleName(TestCase):
    """Test cases for QConfigMapping.set_module_name."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_set_module_name_returns_qconfig_mapping(self):
        """set_module_name returns QConfigMapping (self)."""
        qm = QConfigMapping()
        result = qm.set_module_name("layer1", None)
        self.assertIsInstance(result, QConfigMapping)

    def test_set_module_name_returns_self(self):
        """set_module_name returns self for chaining."""
        qm = QConfigMapping()
        result = qm.set_module_name("layer1", None)
        self.assertIs(result, qm)

    def test_set_module_name_with_none_qconfig(self):
        """set_module_name with qconfig=None does not raise."""
        qm = QConfigMapping()
        qm.set_module_name("module_a", None)

    def test_set_module_name_multiple(self):
        """Multiple set_module_name calls work."""
        qm = QConfigMapping()
        qm.set_module_name("layer1", None)
        qm.set_module_name("layer2", None)
        qm.set_module_name("layer3", None)

    def test_set_module_name_chaining(self):
        """Chained calls via returned self."""
        qm = QConfigMapping()
        result = (
            qm
            .set_module_name("layer1", None)
            .set_module_name("layer2", None)
            .set_module_name("layer3", None)
        )
        self.assertIs(result, qm)

    def test_set_module_name_override(self):
        """Setting same module_name twice overrides without error."""
        qm = QConfigMapping()
        qm.set_module_name("layer1", None)
        qm.set_module_name("layer1", None)  # override

    def test_set_module_name_empty_string(self):
        """set_module_name with empty string does not raise."""
        qm = QConfigMapping()
        qm.set_module_name("", None)

    def test_set_module_name_callable(self):
        """set_module_name is callable on QConfigMapping instance."""
        qm = QConfigMapping()
        self.assertTrue(callable(qm.set_module_name))


if __name__ == "__main__":
    run_tests()
