#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "vllm"
    / "distributed"
    / "kv_transfer"
    / "kv_connector"
    / "v1"
    / "curvine"
    / "store.py"
)


def load_store_module():
    if not MODULE_PATH.exists():
        raise AssertionError(f"Missing module under test: {MODULE_PATH}")

    spec = importlib.util.spec_from_file_location("curvine_store", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from {MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["curvine_store"] = module
    spec.loader.exec_module(module)
    return module


class TestCurvineStore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.block_key = "block-0001"
        self.payload = b"kvblk-payload"

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_client(self):
        module = load_store_module()
        return module, module.PosixCurvineStoreClient(
            root_dir=self.root,
            model_id="meta-llama/Llama-3.2-1B",
            tp_rank=3,
            kv_group_id=1,
        )

    def test_build_block_path_uses_namespace_and_hash_prefix(self):
        module, client = self.make_client()

        block_path = client.build_block_path(self.block_key)
        expected_prefix = hashlib.sha256(self.block_key.encode("utf-8")).hexdigest()[:2]

        self.assertEqual(
            block_path.relative_to(self.root),
            Path("meta-llama_Llama-3.2-1B") / "tp-3" / "kg-1" / expected_prefix / "block-0001.kvblk",
        )
        self.assertTrue(block_path.name.endswith(module.KVBLK_FILE_SUFFIX))

    def test_write_and_read_block_round_trip(self):
        _, client = self.make_client()

        client.write_block(self.block_key, self.payload)

        self.assertTrue(client.exists(self.block_key))
        self.assertEqual(client.read_block(self.block_key), self.payload)

    def test_batch_exists_preserves_input_order(self):
        _, client = self.make_client()
        client.write_block("block-a", b"a")
        client.write_block("block-c", b"c")

        exists = client.batch_exists(["block-a", "block-b", "block-c"])

        self.assertEqual(exists, [True, False, True])

    def test_batch_read_returns_exception_for_missing_block(self):
        module, client = self.make_client()
        client.write_block("block-a", b"a")

        items = client.batch_read(["block-a", "block-missing"])

        self.assertEqual(items[0], b"a")
        self.assertIsInstance(items[1], module.BlockNotFoundError)

    def test_write_block_does_not_leave_temp_files(self):
        _, client = self.make_client()

        client.write_block(self.block_key, self.payload)
        block_dir = client.build_block_path(self.block_key).parent
        leftovers = sorted(p.name for p in block_dir.iterdir() if p.suffix == ".tmp")

        self.assertEqual(leftovers, [])

    def test_delete_block_removes_existing_file(self):
        module, client = self.make_client()
        client.write_block(self.block_key, self.payload)

        client.delete_block(self.block_key)

        self.assertFalse(client.exists(self.block_key))
        with self.assertRaises(module.BlockNotFoundError):
            client.read_block(self.block_key)


if __name__ == "__main__":
    unittest.main()
