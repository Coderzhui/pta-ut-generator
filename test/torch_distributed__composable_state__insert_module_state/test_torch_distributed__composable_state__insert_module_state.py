# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._composable_state._insert_module_state
API 名称：torch.distributed._composable_state._insert_module_state
API 签名：_insert_module_state(module: nn.Module, state: _State) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | N/A                                                          | N/A                                            |
| 枚举选项         | N/A                                                          | N/A                                            |
| 参数类型         | Module + _State                                              | 已覆盖                                         |
| 传参与不传参     | N/A                                                          | N/A                                            |
| 等价类/边界值    | 多次插入同一 module 应 assert                                | 已覆盖                                         |
| 正常传参场景     | _get_module_state 可取回                                     | 已覆盖                                         |
| 异常传参场景     | 重复插入                                                     | 已覆盖 assert                                  |

未覆盖项及原因：
- 无

注意：依赖全局映射副作用，每用例使用独立 Module 实例。
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

from torch.distributed._composable_state import (
    _get_module_state,
    _insert_module_state,
    _State,
)


class TestInsertModuleState(TestCase):
    def test_insert_and_get(self):
        m = nn.Linear(1, 1)
        s = _State()
        _insert_module_state(m, s)
        got = _get_module_state(m)
        self.assertIs(got, s)

    def test_insert_duplicate_raises(self):
        m = nn.Conv2d(1, 1, 1)
        _insert_module_state(m, _State())
        with self.assertRaises(AssertionError):
            _insert_module_state(m, _State())


if __name__ == "__main__":
    run_tests()
