# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._composable.contract.contract 装饰器行为
API 名称：torch.distributed._composable.contract.contract
API 签名：contract(state_cls=_State) -> 装饰器，装饰 (module, *args) -> module

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | state_cls 默认与自定义                                       | 已覆盖默认 _State                              |
| 枚举选项         | N/A                                                          | N/A                                            |
| 参数类型         | nn.Module / list[nn.Module]                                  | 已覆盖 Module 与 list                          |
| 传参与不传参     | @contract() 无参                                             | 已覆盖                                         |
| 等价类/边界值    | 单子模块、多子模块列表                                       | 已覆盖                                         |
| 正常传参场景     | 装饰后具备 .state(module)；返回仍为 Module                   | 已覆盖                                         |
| 异常传参场景     | N/A                                                          | 未覆盖                                         |

未覆盖项及原因：
- FQN 破坏类异常：构造成本高，未覆盖

注意：本测试仅验证 API 结构与状态挂载，不做数值校验。
"""

import unittest

import torch
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

from torch.distributed._composable.contract import contract
from torch.distributed._composable_state import _State


class _CustomState(_State):
    pass


class TestDistributedContract(TestCase):
    def test_contract_default_state_module(self):
        @contract()
        def api(m: nn.Module) -> nn.Module:
            api.state(m).tag = 1
            return m

        m = nn.Linear(4, 4)
        out = api(m)
        self.assertIs(out, m)
        st = api.state(m)
        self.assertIsInstance(st, _State)
        self.assertEqual(getattr(st, "tag", None), 1)

    def test_contract_custom_state_cls(self):
        @contract(_CustomState)
        def api2(m: nn.Module) -> nn.Module:
            self.assertIsInstance(api2.state(m), _CustomState)
            return m

        m = nn.Linear(2, 2)
        api2(m)
        self.assertIsInstance(api2.state(m), _CustomState)

    def test_contract_list_modules(self):
        @contract()
        def api3(modules):
            return modules

        a = nn.Linear(3, 3)
        b = nn.Linear(3, 3)
        out = api3([a, b])
        self.assertIsInstance(out, (list, tuple))


if __name__ == "__main__":
    run_tests()
