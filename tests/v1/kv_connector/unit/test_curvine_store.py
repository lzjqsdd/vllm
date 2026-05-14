#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path, PurePosixPath
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

    def test_store_identity_matches_layout_fields(self):
        _, client = self.make_client()
        ident = client.store_identity
        self.assertEqual(ident.model_id, "meta-llama_Llama-3.2-1B")
        self.assertEqual(ident.tp_rank, 3)
        self.assertEqual(ident.kv_group_id, 1)

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


class FakeCurvineSdk:
    """In-memory FS surface used by NativeCurvineStoreClient contract tests."""

    def __init__(self) -> None:
        self._files: dict[str, bytes] = {}

    def path_exists(self, path: str) -> bool:
        return path in self._files

    def mkdir(self, path: str, create_parents: bool) -> None:  # noqa: ARG002
        return

    def rm(self, path: str, recursive: bool = False) -> None:  # noqa: ARG002
        self._files.pop(path, None)

    def rename(self, src: str, dst: str) -> None:
        if src in self._files:
            self._files[dst] = self._files.pop(src)

    def create(self, path: str, overwrite: bool) -> FakeCurvineSdk._Writer:  # noqa: ARG002
        return FakeCurvineSdk._Writer(self, path)

    class _Writer:
        def __init__(self, outer: "FakeCurvineSdk", path: str) -> None:
            self._outer = outer
            self._path = path
            self._buf = bytearray()

        def write(self, data: bytes) -> None:
            self._buf.extend(data)

        def close(self) -> None:
            self._outer._files[self._path] = bytes(self._buf)

    def read_range(self, path: str, offset: int, length: int) -> bytes:
        data = self._files[path]
        if length == -1:
            return data[offset:]
        return data[offset : offset + length]


class TestCurvineNativeStore(unittest.TestCase):
    def setUp(self):
        self.block_key = "block-0001"
        self.payload = b"kvblk-payload"
        self.kv_root = PurePosixPath("/curvine_kv")
        self.model_id = "meta-llama/Llama-3.2-1B"

    def make_native_client(self, fake: FakeCurvineSdk):
        module = load_store_module()
        client = module.NativeCurvineStoreClient(
            config_path="/no/cluster/required/for/fake",
            root_prefix=str(self.kv_root),
            model_id=self.model_id,
            tp_rank=3,
            kv_group_id=1,
            cv_client=fake,
        )
        return module, client

    def test_native_matches_posix_relative_layout(self):
        fake = FakeCurvineSdk()
        module_posix = load_store_module()
        posix = module_posix.PosixCurvineStoreClient(
            root_dir="/tmp-unused",
            model_id=self.model_id,
            tp_rank=3,
            kv_group_id=1,
        )
        _, native = self.make_native_client(fake)
        posix_rel = posix.build_block_path(self.block_key).relative_to(Path("/tmp-unused"))
        native_rel = native.build_block_path(self.block_key).relative_to(self.kv_root)
        self.assertEqual(native_rel.as_posix(), posix_rel.as_posix())

    def test_native_round_trip(self):
        fake = FakeCurvineSdk()
        _, client = self.make_native_client(fake)
        client.write_block(self.block_key, self.payload)
        self.assertTrue(client.exists(self.block_key))
        self.assertEqual(client.read_block(self.block_key), self.payload)

    def test_native_batch_read_missing(self):
        module, client = self.make_native_client(FakeCurvineSdk())
        client.write_block("block-a", b"a")
        items = client.batch_read(["block-a", "block-missing"])
        self.assertEqual(items[0], b"a")
        self.assertIsInstance(items[1], module.BlockNotFoundError)

    def test_native_no_tmp_suffix_after_write(self):
        fake = FakeCurvineSdk()
        _, client = self.make_native_client(fake)
        client.write_block(self.block_key, self.payload)
        tmp_left = [k for k in fake._files if k.endswith(".tmp")]
        self.assertEqual(tmp_left, [])

    def test_make_store_client_native_requires_config_path(self):
        module = load_store_module()
        with self.assertRaises(ValueError):
            module.make_curvine_store_client(
                {"curvine_backend": "native"},
                default_model_id="m",
            )

    def test_make_store_client_unknown_backend(self):
        module = load_store_module()
        with self.assertRaises(ValueError):
            module.make_curvine_store_client(
                {"curvine_backend": "invalid"},
                default_model_id="m",
            )


if __name__ == "__main__":
    unittest.main()
