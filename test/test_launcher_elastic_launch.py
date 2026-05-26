# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.launcher.elastic_launch 接口功能正确性
API 名称：torch.distributed.launcher.elastic_launch
API 签名：class elastic_launch:
             def __init__(config: LaunchConfig, entrypoint: Callable | str | None)
             def __call__(self, *args) -> dict[int, Any]

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | entrypoint=None vs Callable vs str                           | 已覆盖                                         |
| 枚举选项         | N/A（无枚举参数）                                            | N/A                                            |
| 参数类型         | config=LaunchConfig; entrypoint=Callable/str/None            | 已覆盖                                         |
| 传参与不传参     | __init__ 必传 config 和 entrypoint                           | 已覆盖                                         |
| 等价类/边界值    | entrypoint 三种类型                                          | 已覆盖                                         |
| 正常传参场景     | 构造实例、属性访问、类型检查                                  | 已覆盖                                         |
| 异常传参场景     | 无稳定异常路径（构造不做校验）                               | 未覆盖：构造函数不校验参数                     |
| 混合设备类型     | 单对象构造，不涉及多 Tensor 输入                             | 未覆盖：不适用                                 |

未覆盖项及原因：
- 异常传参场景：elastic_launch.__init__ 不做严格参数校验
- 混合设备类型：API 为单对象构造，不涉及多 Tensor 输入
- __call__ 未测试：会启动真实 elastic agent 进程，不适合在 UT 中调用

注意：本测试仅验证功能正确性（构造不报错、属性类型正确），
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

from torch.distributed.launcher import elastic_launch, LaunchConfig


def _dummy_entrypoint():
    return 0


class TestLauncherElasticLaunch(TestCase):
    """Test cases for torch.distributed.launcher.elastic_launch."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def _make_config(self):
        return LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1)

    def test_construction_with_callable(self):
        """Construct with callable entrypoint."""
        config = self._make_config()
        launcher = elastic_launch(config, _dummy_entrypoint)
        self.assertIsNotNone(launcher)

    def test_construction_with_string(self):
        """Construct with string entrypoint."""
        config = self._make_config()
        launcher = elastic_launch(config, "train.py")
        self.assertIsNotNone(launcher)

    def test_construction_with_none(self):
        """Construct with None entrypoint."""
        config = self._make_config()
        launcher = elastic_launch(config, None)
        self.assertIsNotNone(launcher)

    def test_stores_config(self):
        """_config attribute stores LaunchConfig."""
        config = self._make_config()
        launcher = elastic_launch(config, None)
        self.assertIs(launcher._config, config)

    def test_stores_callable_entrypoint(self):
        """_entrypoint attribute stores callable."""
        config = self._make_config()
        launcher = elastic_launch(config, _dummy_entrypoint)
        self.assertIs(launcher._entrypoint, _dummy_entrypoint)

    def test_stores_string_entrypoint(self):
        """_entrypoint attribute stores string."""
        config = self._make_config()
        launcher = elastic_launch(config, "train.py")
        self.assertEqual(launcher._entrypoint, "train.py")

    def test_stores_none_entrypoint(self):
        """_entrypoint attribute stores None."""
        config = self._make_config()
        launcher = elastic_launch(config, None)
        self.assertIsNone(launcher._entrypoint)

    def test_is_callable(self):
        """Instance is callable (has __call__)."""
        config = self._make_config()
        launcher = elastic_launch(config, None)
        self.assertTrue(callable(launcher))

    def test_type(self):
        """Instance type check."""
        config = self._make_config()
        launcher = elastic_launch(config, None)
        self.assertIsInstance(launcher, elastic_launch)


if __name__ == "__main__":
    run_tests()
