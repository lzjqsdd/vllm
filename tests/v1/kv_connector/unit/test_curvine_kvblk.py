#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path
import sys
import unittest

MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "vllm"
    / "distributed"
    / "kv_transfer"
    / "kv_connector"
    / "v1"
    / "curvine"
    / "kvblk.py"
)
SPEC = importlib.util.spec_from_file_location("curvine_kvblk", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Unable to load module spec from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["curvine_kvblk"] = MODULE
SPEC.loader.exec_module(MODULE)

HEADER_LEN_V1 = MODULE.HEADER_LEN_V1
CacheLayout = MODULE.CacheLayout
CanonicalOrder = MODULE.CanonicalOrder
DTypeCode = MODULE.DTypeCode
KvblkFormatError = MODULE.KvblkFormatError
KvblkHeader = MODULE.KvblkHeader
PayloadCodec = MODULE.PayloadCodec
TensorFormat = MODULE.TensorFormat
deserialize_kvblk = MODULE.deserialize_kvblk
serialize_kvblk = MODULE.serialize_kvblk


def make_header() -> KvblkHeader:
    return KvblkHeader(
        version=1,
        flags=0,
        checksum_type=1,
        payload_codec=PayloadCodec.RAW,
        block_key_hash=bytes.fromhex("0123456789abcdeffedcba9876543210"),
        model_id_hash=0x12345678,
        tp_rank=0,
        kv_group_id=0,
        cache_layout=CacheLayout.NHD,
        dtype_code=DTypeCode.BF16,
        block_size_tokens=16,
        page_size_bytes=128,
        layer_count=2,
        num_kv_heads=8,
        head_size=16,
        tensor_format=TensorFormat.FULL_KV_SEPARATE,
        canonical_order=CanonicalOrder.LAYER_MAJOR_K_THEN_V,
        created_at_unix_ms=1710000000000,
    )


class TestCurvineKvblk(unittest.TestCase):
    def test_round_trip_serialization_preserves_header_and_payload(self):
        header = make_header()
        payload = bytes(range(64))

        blob = serialize_kvblk(header, payload)
        decoded_header, decoded_payload = deserialize_kvblk(blob)

        self.assertEqual(len(blob), HEADER_LEN_V1 + len(payload))
        self.assertEqual(decoded_payload, payload)
        self.assertEqual(decoded_header.header_len, HEADER_LEN_V1)
        self.assertEqual(decoded_header.body_len, len(payload))
        self.assertEqual(decoded_header.payload_codec, PayloadCodec.RAW)
        self.assertEqual(decoded_header.cache_layout, CacheLayout.NHD)
        self.assertEqual(decoded_header.dtype_code, DTypeCode.BF16)
        self.assertEqual(
            decoded_header.tensor_format, TensorFormat.FULL_KV_SEPARATE
        )
        self.assertEqual(
            decoded_header.canonical_order,
            CanonicalOrder.LAYER_MAJOR_K_THEN_V,
        )

    def test_deserialize_rejects_invalid_magic(self):
        header = make_header()
        payload = bytes(range(16))
        blob = bytearray(serialize_kvblk(header, payload))
        blob[0:4] = b"FAIL"

        with self.assertRaisesRegex(KvblkFormatError, "magic"):
            deserialize_kvblk(bytes(blob))

    def test_deserialize_rejects_corrupted_header_crc(self):
        header = make_header()
        payload = bytes(range(16))
        blob = bytearray(serialize_kvblk(header, payload))
        blob[20] ^= 0xFF

        with self.assertRaisesRegex(KvblkFormatError, "header crc"):
            deserialize_kvblk(bytes(blob))

    def test_deserialize_rejects_corrupted_body_crc(self):
        header = make_header()
        payload = bytes(range(16))
        blob = bytearray(serialize_kvblk(header, payload))
        blob[-1] ^= 0xFF

        with self.assertRaisesRegex(KvblkFormatError, "body crc"):
            deserialize_kvblk(bytes(blob))

    def test_deserialize_rejects_truncated_body(self):
        header = make_header()
        payload = bytes(range(32))
        blob = serialize_kvblk(header, payload)

        with self.assertRaisesRegex(KvblkFormatError, "truncated"):
            deserialize_kvblk(blob[:-3])


if __name__ == "__main__":
    unittest.main()
