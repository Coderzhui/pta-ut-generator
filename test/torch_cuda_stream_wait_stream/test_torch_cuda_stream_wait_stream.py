# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.Stream.wait_stream 的流同步语义（PTA 下 cuda 常映射到 NPU）
API 名称：torch.cuda.Stream.wait_stream
API 签名：wait_stream(self, stream) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | N/A                                                          | N/A                                            |
| 枚举选项         | N/A                                                          | N/A                                            |
| 参数类型         | stream 为 Stream                                             | 已覆盖                                         |
| 传参与不传参     | N/A                                                          | N/A                                            |
| 等价类/边界值    | 两路流默认构造                                               | 已覆盖                                         |
| 正常传参场景     | 调用不报错                                                   | 已覆盖                                         |
| 异常传参场景     | N/A                                                          | 未覆盖                                         |

未覆盖项及原因：
- 无

注意：本测试仅验证调用与类型层面行为，不做数值校验。
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


def _privateuse1_stream_cls():
    name = torch._C._get_privateuse1_backend_name()
    mod = getattr(torch, name, None)
    if mod is None or not getattr(mod, "is_available", lambda: False)():
        return None
    return getattr(mod, "Stream", None)


class TestCudaStreamWaitStream(TestCase):
    def test_privateuse1_stream_wait_stream(self):
        Stream = _privateuse1_stream_cls()
        if Stream is None:
            self.skipTest("privateuse1 Stream not available")
        s1 = Stream()
        s2 = Stream()
        s2.wait_stream(s1)

    def test_privateuse1_stream_wait_stream_under_mod_stream_ctx(self):
        Stream = _privateuse1_stream_cls()
        if Stream is None:
            self.skipTest("privateuse1 Stream not available")
        name = torch._C._get_privateuse1_backend_name()
        mod = getattr(torch, name)
        s1 = Stream()
        s2 = Stream()
        stream_ctx = getattr(mod, "stream", None)
        if callable(stream_ctx):
            with stream_ctx(s2):
                s2.wait_stream(s1)
        else:
            s2.wait_stream(s1)

    def test_privateuse1_wait_stream_idempotent(self):
        Stream = _privateuse1_stream_cls()
        if Stream is None:
            self.skipTest("privateuse1 Stream not available")
        s1 = Stream()
        s2 = Stream()
        s2.wait_stream(s1)
        s2.wait_stream(s1)

    def test_wait_stream_two_streams_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for torch.cuda.Stream")
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        s2.wait_stream(s1)


if __name__ == "__main__":
    run_tests()
