# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._composable.contract._get_registry
API 名称：torch.distributed._composable.contract._get_registry
API 签名：_get_registry(module: nn.Module) -> Optional[dict[str, RegistryItem]]

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 未应用 composable 时为 None                                 | 已覆盖                                         |
| 枚举选项         | N/A                                                          | N/A                                            |
| 参数类型         | nn.Module                                                    | 已覆盖                                         |
| 传参与不传参     | N/A                                                          | N/A                                            |
| 等价类/边界值    | 应用前后                                                     | 已覆盖                                         |
| 正常传参场景     | 应用 @contract 后返回 dict                                   | 已覆盖                                         |
| 异常传参场景     | N/A                                                          | 未覆盖                                         |

未覆盖项及原因：
- 无

注意：仅验证返回类型与键存在性，不做数值校验。
"""

import unittest

import torch.nn as nn

try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    pass

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

from torch.distributed._composable.contract import _get_registry, contract


class TestGetRegistry(TestCase):
    def test_get_registry_none_before_apply(self):
        m = nn.Linear(2, 2)
        self.assertIsNone(_get_registry(m))

    def test_get_registry_after_contract(self):
        @contract()
        def my_api(mod: nn.Module) -> nn.Module:
            return mod

        m = nn.Linear(2, 2)
        my_api(m)
        reg = _get_registry(m)
        self.assertIsNotNone(reg)
        self.assertIn("my_api", reg)


if __name__ == "__main__":
    run_tests()
