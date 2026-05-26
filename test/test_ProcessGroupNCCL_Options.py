# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.ProcessGroupNCCL.Options 接口功能正确性
API 名称：torch.distributed.ProcessGroupNCCL.Options
API 签名：class Options(Backend.Options):
             config: ProcessGroupNCCL.NCCLConfig
             is_high_priority_stream: bool
             split_from: ProcessGroupNCCL
             split_color: int
             def __init__(is_high_priority_stream: bool = False)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                       |
|------------------|--------------------------------------------------------------|--------------------------------|
| 基础构造         | 默认参数构造与显式参数构造不报错、返回类型正确               | 已覆盖                         |
| 参数枚举/边界值  | is_high_priority_stream True/False 两种取值                  | 已覆盖                         |
| 默认值验证       | 各属性默认值与文档一致                                       | 已覆盖                         |
| 属性访问         | config / is_high_priority_stream / split_from / split_color | 已覆盖                         |
| 属性类型         | 各属性返回类型与声明一致                                     | 已覆盖                         |
| 继承关系         | Options 继承自 Backend.Options，具备 backend 等基类属性      | 已覆盖                         |
| 异常路径         | ProcessGroupNCCL 不可用时优雅报错                           | 已覆盖                         |
| 混合设备类型     | 单进程配置类，不涉及多 Tensor 设备混合                       | 未覆盖：API 为纯配置类         |

未覆盖项及原因：
- 混合设备类型：ProcessGroupNCCL.Options 为单进程配置类，不涉及多 Tensor 输入及设备迁移
- split_from 属性赋值：需要已初始化的 ProcessGroupNCCL 实例，单进程环境无法构造

注意：本测试仅验证功能正确性（构造不报错、属性类型/默认值符合预期），
     不做精度和数值正确性校验。
"""
import unittest
import torch
import torch_npu  # noqa: F401

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


# ProcessGroupNCCL may not be available on all builds
_ProcessGroupNCCL_available = True
_ProcessGroupNCCL = None
_ProcessGroupNCCL_Options = None
try:
    from torch._C._distributed_c10d import ProcessGroupNCCL as _ProcessGroupNCCL
    _ProcessGroupNCCL_Options = _ProcessGroupNCCL.Options
except ImportError:
    _ProcessGroupNCCL_available = False


class TestProcessGroupNCCLOptions(TestCase):
    """Test cases for torch.distributed.ProcessGroupNCCL.Options."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    @unittest.skipIf(not _ProcessGroupNCCL_available,
                     "ProcessGroupNCCL is not available in this build")
    def test_default_construction(self):
        """Options() with no arguments uses default is_high_priority_stream=False."""
        opts = _ProcessGroupNCCL_Options()
        self.assertIsInstance(opts, _ProcessGroupNCCL_Options)

    @unittest.skipIf(not _ProcessGroupNCCL_available,
                     "ProcessGroupNCCL is not available in this build")
    def test_construction_with_true(self):
        """Options(is_high_priority_stream=True) constructs without error."""
        opts = _ProcessGroupNCCL_Options(is_high_priority_stream=True)
        self.assertIsInstance(opts, _ProcessGroupNCCL_Options)

    @unittest.skipIf(not _ProcessGroupNCCL_available,
                     "ProcessGroupNCCL is not available in this build")
    def test_construction_with_false(self):
        """Options(is_high_priority_stream=False) constructs without error."""
        opts = _ProcessGroupNCCL_Options(is_high_priority_stream=False)
        self.assertIsInstance(opts, _ProcessGroupNCCL_Options)

    @unittest.skipIf(not _ProcessGroupNCCL_available,
                     "ProcessGroupNCCL is not available in this build")
    def test_default_is_high_priority_stream(self):
        """Default is_high_priority_stream should be False."""
        opts = _ProcessGroupNCCL_Options()
        self.assertFalse(opts.is_high_priority_stream)

    @unittest.skipIf(not _ProcessGroupNCCL_available,
                     "ProcessGroupNCCL is not available in this build")
    def test_is_high_priority_stream_set_to_true(self):
        """Explicitly setting is_high_priority_stream=True should be reflected."""
        opts = _ProcessGroupNCCL_Options(is_high_priority_stream=True)
        self.assertTrue(opts.is_high_priority_stream)

    @unittest.skipIf(not _ProcessGroupNCCL_available,
                     "ProcessGroupNCCL is not available in this build")
    def test_is_high_priority_stream_type(self):
        """is_high_priority_stream should be a bool."""
        opts = _ProcessGroupNCCL_Options()
        self.assertIsInstance(opts.is_high_priority_stream, bool)

    @unittest.skipIf(not _ProcessGroupNCCL_available,
                     "ProcessGroupNCCL is not available in this build")
    def test_config_attribute_exists(self):
        """config attribute should exist and be an NCCLConfig instance."""
        opts = _ProcessGroupNCCL_Options()
        self.assertTrue(hasattr(opts, 'config'))

    @unittest.skipIf(not _ProcessGroupNCCL_available,
                     "ProcessGroupNCCL is not available in this build")
    def test_config_attribute_type(self):
        """config attribute should be of type ProcessGroupNCCL.NCCLConfig."""
        opts = _ProcessGroupNCCL_Options()
        self.assertIsInstance(opts.config, _ProcessGroupNCCL.NCCLConfig)

    @unittest.skipIf(not _ProcessGroupNCCL_available,
                     "ProcessGroupNCCL is not available in this build")
    def test_config_blocking_attribute(self):
        """NCCLConfig should have a 'blocking' attribute of type int."""
        opts = _ProcessGroupNCCL_Options()
        self.assertIsInstance(opts.config.blocking, int)

    @unittest.skipIf(not _ProcessGroupNCCL_available,
                     "ProcessGroupNCCL is not available in this build")
    def test_config_cga_cluster_size_attribute(self):
        """NCCLConfig should have a 'cga_cluster_size' attribute of type int."""
        opts = _ProcessGroupNCCL_Options()
        self.assertIsInstance(opts.config.cga_cluster_size, int)

    @unittest.skipIf(not _ProcessGroupNCCL_available,
                     "ProcessGroupNCCL is not available in this build")
    def test_config_min_ctas_attribute(self):
        """NCCLConfig should have a 'min_ctas' attribute of type int."""
        opts = _ProcessGroupNCCL_Options()
        self.assertIsInstance(opts.config.min_ctas, int)

    @unittest.skipIf(not _ProcessGroupNCCL_available,
                     "ProcessGroupNCCL is not available in this build")
    def test_config_max_ctas_attribute(self):
        """NCCLConfig should have a 'max_ctas' attribute of type int."""
        opts = _ProcessGroupNCCL_Options()
        self.assertIsInstance(opts.config.max_ctas, int)

    @unittest.skipIf(not _ProcessGroupNCCL_available,
                     "ProcessGroupNCCL is not available in this build")
    def test_split_color_attribute_exists(self):
        """split_color attribute should exist."""
        opts = _ProcessGroupNCCL_Options()
        self.assertTrue(hasattr(opts, 'split_color'))

    @unittest.skipIf(not _ProcessGroupNCCL_available,
                     "ProcessGroupNCCL is not available in this build")
    def test_split_color_attribute_type(self):
        """split_color attribute should be of type int."""
        opts = _ProcessGroupNCCL_Options()
        self.assertIsInstance(opts.split_color, int)

    @unittest.skipIf(not _ProcessGroupNCCL_available,
                     "ProcessGroupNCCL is not available in this build")
    def test_split_from_attribute_exists(self):
        """split_from attribute should exist (default may be None)."""
        opts = _ProcessGroupNCCL_Options()
        self.assertTrue(hasattr(opts, 'split_from'))

    @unittest.skipIf(not _ProcessGroupNCCL_available,
                     "ProcessGroupNCCL is not available in this build")
    def test_is_high_priority_stream_mutable(self):
        """is_high_priority_stream can be modified after construction."""
        opts = _ProcessGroupNCCL_Options()
        opts.is_high_priority_stream = True
        self.assertTrue(opts.is_high_priority_stream)
        opts.is_high_priority_stream = False
        self.assertFalse(opts.is_high_priority_stream)

    @unittest.skipIf(not _ProcessGroupNCCL_available,
                     "ProcessGroupNCCL is not available in this build")
    def test_split_color_mutable(self):
        """split_color can be modified after construction."""
        opts = _ProcessGroupNCCL_Options()
        original = opts.split_color
        opts.split_color = 42
        self.assertEqual(opts.split_color, 42)
        opts.split_color = original

    @unittest.skipIf(not _ProcessGroupNCCL_available,
                     "ProcessGroupNCCL is not available in this build")
    def test_inherits_backend_options(self):
        """Options should inherit from Backend.Options with 'backend' property."""
        from torch._C._distributed_c10d import Backend
        opts = _ProcessGroupNCCL_Options()
        self.assertIsInstance(opts, Backend.Options)

    @unittest.skipIf(not _ProcessGroupNCCL_available,
                     "ProcessGroupNCCL is not available in this build")
    def test_backend_property_type(self):
        """The inherited 'backend' property should return a string."""
        opts = _ProcessGroupNCCL_Options()
        backend_val = opts.backend
        self.assertIsInstance(backend_val, str)

    def test_import_unavailable_graceful(self):
        """When ProcessGroupNCCL is unavailable, import should be handled gracefully.

        This test always passes: if ProcessGroupNCCL IS available it verifies
        the import succeeded; if not, it verifies the skip flag is set.
        """
        if _ProcessGroupNCCL_available:
            self.assertIsNotNone(_ProcessGroupNCCL_Options)
        else:
            self.assertIsNone(_ProcessGroupNCCL_Options)


if __name__ == "__main__":
    run_tests()
