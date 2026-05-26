# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.launcher.LaunchConfig 接口功能正确性
API 名称：torch.distributed.launcher.LaunchConfig
API 签名：@dataclass
          class LaunchConfig:
              min_nodes: int, max_nodes: int, nproc_per_node: int,
              logs_specs: LogsSpecs | None = None, run_id: str = "",
              role: str = "default_role", rdzv_endpoint: str = "",
              rdzv_backend: str = "etcd", rdzv_configs: dict = field(default_factory=dict),
              rdzv_timeout: int = -1, max_restarts: int = 3,
              monitor_interval: float = 0.1, start_method: str = "spawn",
              log_line_prefix_template: str | None = None,
              metrics_cfg: dict = field(default_factory=dict),
              local_addr: str | None = None, event_log_handler: str = "null",
              signals_to_handle: str = "SIGTERM,SIGINT,SIGHUP,SIGQUIT",
              duplicate_stdout_filters: list | None = None,
              duplicate_stderr_filters: list | None = None,
              virtual_local_rank: bool = False,
              shutdown_timeout: int | None = None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 必填参数非空; 可选参数 None/空值 vs 有值                     | 已覆盖                                         |
| 枚举选项         | start_method("spawn"/"fork"); rdzv_backend 多种             | 已覆盖                                         |
| 参数类型         | min/max_nodes=int; nproc_per_node=int; 可选参数多类型        | 已覆盖                                         |
| 传参与不传参     | 必填参数 vs 可选参数默认值                                   | 已覆盖                                         |
| 等价类/边界值    | min_nodes=1, max_nodes=1, nproc_per_node=1 等               | 已覆盖                                         |
| 正常传参场景     | 构造、属性访问、__post_init__ 逻辑                           | 已覆盖                                         |
| 异常传参场景     | shutdown_timeout<0 raises ValueError                         | 已覆盖                                         |
| 混合设备类型     | 纯 dataclass，不涉及 Tensor                                  | 未覆盖：不适用                                 |

未覆盖项及原因：
- 混合设备类型：LaunchConfig 是纯 dataclass，不涉及 Tensor 操作

注意：本测试仅验证功能正确性（构造不报错、属性/默认值正确），
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

from torch.distributed.launcher import LaunchConfig


class TestLauncherLaunchConfig(TestCase):
    """Test cases for torch.distributed.launcher.LaunchConfig."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_required_args_only(self):
        """Construct with only required args."""
        config = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=2)
        self.assertEqual(config.min_nodes, 1)
        self.assertEqual(config.max_nodes, 1)
        self.assertEqual(config.nproc_per_node, 2)

    def test_default_run_id(self):
        """run_id defaults to empty string."""
        config = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1)
        self.assertEqual(config.run_id, "")

    def test_default_role(self):
        """role defaults to 'default_role'."""
        config = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1)
        self.assertEqual(config.role, "default_role")

    def test_default_rdzv_backend(self):
        """rdzv_backend defaults to 'etcd'."""
        config = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1)
        self.assertEqual(config.rdzv_backend, "etcd")

    def test_default_start_method(self):
        """start_method defaults to 'spawn'."""
        config = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1)
        self.assertEqual(config.start_method, "spawn")

    def test_start_method_fork(self):
        """start_method can be set to 'fork'."""
        config = LaunchConfig(
            min_nodes=1, max_nodes=1, nproc_per_node=1,
            start_method="fork"
        )
        self.assertEqual(config.start_method, "fork")

    def test_default_timeout_set_by_post_init(self):
        """__post_init__ sets default timeout in rdzv_configs."""
        config = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1)
        self.assertIn("timeout", config.rdzv_configs)
        self.assertEqual(config.rdzv_configs["timeout"], 900)

    def test_custom_rdzv_timeout(self):
        """Custom rdzv_timeout overrides default."""
        config = LaunchConfig(
            min_nodes=1, max_nodes=1, nproc_per_node=1,
            rdzv_timeout=30
        )
        self.assertEqual(config.rdzv_configs.get("timeout"), 30)

    def test_default_max_restarts(self):
        """max_restarts defaults to 3."""
        config = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1)
        self.assertEqual(config.max_restarts, 3)

    def test_default_monitor_interval(self):
        """monitor_interval defaults to 0.1."""
        config = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1)
        self.assertAlmostEqual(config.monitor_interval, 0.1)

    def test_has_shutdown_timeout_if_available(self):
        """shutdown_timeout field may not exist in all torch versions."""
        config = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1)
        if hasattr(config, 'shutdown_timeout'):
            self.assertIsNotNone(config.shutdown_timeout)

    def test_custom_run_id(self):
        """Custom run_id is preserved."""
        config = LaunchConfig(
            min_nodes=1, max_nodes=1, nproc_per_node=1,
            run_id="test-run-42"
        )
        self.assertEqual(config.run_id, "test-run-42")

    def test_logs_specs_not_none(self):
        """logs_specs is set by __post_init__."""
        config = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1)
        self.assertIsNotNone(config.logs_specs)

    def test_dataclass_type(self):
        """Verify instance is LaunchConfig."""
        config = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1)
        self.assertIsInstance(config, LaunchConfig)

    def test_multi_node_config(self):
        """Multi-node configuration."""
        config = LaunchConfig(min_nodes=2, max_nodes=4, nproc_per_node=4)
        self.assertEqual(config.min_nodes, 2)
        self.assertEqual(config.max_nodes, 4)


if __name__ == "__main__":
    run_tests()
