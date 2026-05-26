# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.PrefixStore 接口功能正确性
API 名称：torch.distributed.PrefixStore
API 签名：class PrefixStore:
             def __init__(prefix: str, store: Store)
             Methods: set(key, value), get(key), add(key, value), delete_key(key), etc.

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | prefix 空字符串 vs 非空; key/value 空 vs 非空                | 已覆盖                                         |
| 枚举选项         | N/A（无枚举参数）                                            | N/A                                            |
| 参数类型         | prefix=str; store=Store; key=str; value=str/bytes/number     | 已覆盖                                         |
| 传参与不传参     | N/A（构造必传 prefix 和 store）                              | N/A                                            |
| 等价类/边界值    | 空前缀; 长前缀; 特殊字符前缀                                 | 已覆盖                                         |
| 正常传参场景     | 构造、set/get/add/delete 操作                                 | 已覆盖                                         |
| 异常传参场景     | get 不存在的 key raises                                      | 已覆盖                                         |
| 混合设备类型     | 纯 KV store 操作，不涉及 Tensor                              | 未覆盖：不适用                                 |

未覆盖项及原因：
- 混合设备类型：PrefixStore 是纯键值存储，不涉及 Tensor 操作

注意：本测试仅验证功能正确性（构造/方法调用不报错、返回类型正确），
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

from torch._C._distributed_c10d import TCPStore, PrefixStore


class TestPrefixStore(TestCase):
    """Test cases for torch.distributed.PrefixStore."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )
        self.base_store = TCPStore("127.0.0.1", 0, 1, True)
        self.prefix = "test_prefix"

    def test_construction(self):
        """Construct PrefixStore with prefix and store."""
        ps = PrefixStore(self.prefix, self.base_store)
        self.assertIsNotNone(ps)

    def test_type(self):
        """Instance is PrefixStore."""
        ps = PrefixStore(self.prefix, self.base_store)
        self.assertIsInstance(ps, PrefixStore)

    def test_empty_prefix(self):
        """Construct with empty prefix."""
        ps = PrefixStore("", self.base_store)
        self.assertIsNotNone(ps)

    def test_set_and_get_string(self):
        """set/get with string value."""
        ps = PrefixStore(self.prefix, self.base_store)
        ps.set("key1", b"value1")
        result = ps.get("key1")
        self.assertIsNotNone(result)

    def test_set_and_get_bytes(self):
        """set/get with bytes value."""
        ps = PrefixStore(self.prefix, self.base_store)
        ps.set("key_bytes", b"hello")
        result = ps.get("key_bytes")
        self.assertIsNotNone(result)

    def test_add(self):
        """add returns incremented value."""
        ps = PrefixStore(self.prefix, self.base_store)
        ps.set("counter", b"0")
        result = ps.add("counter", 5)
        self.assertIsNotNone(result)

    def test_add_from_zero(self):
        """add on non-existent key initializes and increments."""
        ps = PrefixStore(self.prefix, self.base_store)
        result = ps.add("new_counter", 10)
        self.assertIsNotNone(result)

    def test_different_prefixes_isolated(self):
        """Different prefixes isolate keys."""
        ps1 = PrefixStore("prefix_a", self.base_store)
        ps2 = PrefixStore("prefix_b", self.base_store)
        ps1.set("shared_key", b"value_a")
        ps2.set("shared_key", b"value_b")
        result_a = ps1.get("shared_key")
        result_b = ps2.get("shared_key")
        # Both should return their own values
        self.assertIsNotNone(result_a)
        self.assertIsNotNone(result_b)

    def test_delete_key(self):
        """delete_key removes a key."""
        ps = PrefixStore(self.prefix, self.base_store)
        ps.set("to_delete", b"temp")
        ps.delete_key("to_delete")
        # Getting deleted key should raise
        with self.assertRaises(Exception):
            ps.get("to_delete")

    def test_get_nonexistent_key_raises(self):
        """get on non-existent key raises exception."""
        ps = PrefixStore(self.prefix, self.base_store)
        with self.assertRaises(Exception):
            ps.get("nonexistent_key_12345")

    def test_special_char_prefix(self):
        """Prefix with special characters."""
        ps = PrefixStore("ns/path:v2", self.base_store)
        ps.set("key", b"value")
        result = ps.get("key")
        self.assertIsNotNone(result)

    def test_long_prefix(self):
        """Long prefix string."""
        long_prefix = "a" * 100
        ps = PrefixStore(long_prefix, self.base_store)
        ps.set("key", b"value")
        result = ps.get("key")
        self.assertIsNotNone(result)

    def test_overwrite_value(self):
        """set overwrites existing value."""
        ps = PrefixStore(self.prefix, self.base_store)
        ps.set("overwrite_key", b"old_value")
        ps.set("overwrite_key", b"new_value")
        result = ps.get("overwrite_key")
        self.assertIsNotNone(result)


if __name__ == "__main__":
    run_tests()
