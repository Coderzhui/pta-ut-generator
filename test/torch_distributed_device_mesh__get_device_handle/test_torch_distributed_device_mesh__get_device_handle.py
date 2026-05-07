# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.device_mesh._get_device_handle
API 名称：torch.distributed.device_mesh._get_device_handle
API 签名：_get_device_handle(device_type: str = "cuda") -> Optional[module]

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | N/A                                                          | N/A                                            |
| 枚举选项         | N/A                                                          | N/A                                            |
| 参数类型         | str device_type                                              | 已覆盖                                         |
| 传参与不传参     | 默认 "cuda" 与显式传入                                       | 已覆盖                                         |
| 等价类/边界值    | cpu / cuda / privateuse1 名                                  | 已覆盖                                         |
| 正常传参场景     | 返回 getattr(torch, device_type, None)                       | 已覆盖                                         |
| 异常传参场景     | 不存在设备名 → None                                          | 已覆盖                                         |

未覆盖项及原因：
- 无

注意：仅验证返回值与 torch 子模块一致性，不做设备算子校验。
"""

import unittest

import torch

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

import torch.distributed.device_mesh as device_mesh_mod


class TestGetDeviceHandle(TestCase):
    def test_default_cuda(self):
        h = device_mesh_mod._get_device_handle()
        self.assertIs(h, getattr(torch, "cuda", None))

    def test_explicit_cpu(self):
        h = device_mesh_mod._get_device_handle("cpu")
        # cpu 不是子模块时可能为 None，与 getattr 一致
        self.assertIs(h, getattr(torch, "cpu", None))

    def test_privateuse1_name(self):
        name = torch._C._get_privateuse1_backend_name()
        h = device_mesh_mod._get_device_handle(name)
        self.assertIs(h, getattr(torch, name, None))

    def test_unknown_type_none(self):
        h = device_mesh_mod._get_device_handle("__no_such_device_type__")
        self.assertIsNone(h)


if __name__ == "__main__":
    run_tests()
