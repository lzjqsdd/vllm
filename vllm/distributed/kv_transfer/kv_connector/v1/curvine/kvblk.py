# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, replace
from enum import IntEnum

MAGIC_V1 = b"KVBL"
VERSION_V1 = 1
HEADER_LEN_V1 = 128

_HEADER_STRUCT = struct.Struct("<4sHHIHHIIQ16sQIIHHIIIIIHHQ28s")


class KvblkFormatError(ValueError):
    """Raised when kvblk bytes do not match the expected format."""


class ChecksumType(IntEnum):
    """Checksum algorithms supported by kvblk."""

    CRC32 = 1


class PayloadCodec(IntEnum):
    """Payload encoding methods supported by kvblk."""

    RAW = 0


class CacheLayout(IntEnum):
    """Logical KV cache layout identifiers."""

    NHD = 1
    HND = 2
    MLA = 3


class DTypeCode(IntEnum):
    """Supported element types for serialized payload."""

    FP16 = 1
    BF16 = 2
    FP32 = 3
    FP8_E4M3FN = 4
    FP8_E5M2 = 5
    INT8 = 6


class TensorFormat(IntEnum):
    """Semantic interpretation of the payload body."""

    FULL_KV_SEPARATE = 1
    MLA_LINEAR = 2


class CanonicalOrder(IntEnum):
    """Canonical byte ordering inside the payload body."""

    LAYER_MAJOR_K_THEN_V = 1
    LAYER_MAJOR_LINEAR = 2


def _crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


@dataclass(frozen=True, slots=True)
class KvblkHeader:
    """Fixed-size V1 header for PoC kvblk files."""

    version: int = VERSION_V1
    header_len: int = HEADER_LEN_V1
    flags: int = 0
    checksum_type: int = ChecksumType.CRC32
    payload_codec: int = PayloadCodec.RAW
    header_crc32: int = 0
    body_crc32: int = 0
    body_len: int = 0
    block_key_hash: bytes = b"\x00" * 16
    model_id_hash: int = 0
    tp_rank: int = 0
    kv_group_id: int = 0
    cache_layout: int = CacheLayout.NHD
    dtype_code: int = DTypeCode.BF16
    block_size_tokens: int = 0
    page_size_bytes: int = 0
    layer_count: int = 0
    num_kv_heads: int = 0
    head_size: int = 0
    tensor_format: int = TensorFormat.FULL_KV_SEPARATE
    canonical_order: int = CanonicalOrder.LAYER_MAJOR_K_THEN_V
    created_at_unix_ms: int = 0
    reserved: bytes = b"\x00" * 28

    def __post_init__(self) -> None:
        if len(self.block_key_hash) != 16:
            raise KvblkFormatError("block_key_hash must be 16 bytes")
        if len(self.reserved) != 28:
            raise KvblkFormatError("reserved must be 28 bytes")
        if self.version != VERSION_V1:
            raise KvblkFormatError(f"unsupported version: {self.version}")
        if self.header_len != HEADER_LEN_V1:
            raise KvblkFormatError(f"unsupported header_len: {self.header_len}")
        ChecksumType(self.checksum_type)
        PayloadCodec(self.payload_codec)
        CacheLayout(self.cache_layout)
        DTypeCode(self.dtype_code)
        TensorFormat(self.tensor_format)
        CanonicalOrder(self.canonical_order)

    def _pack(self) -> bytes:
        return _HEADER_STRUCT.pack(
            MAGIC_V1,
            self.version,
            self.header_len,
            self.flags,
            self.checksum_type,
            self.payload_codec,
            self.header_crc32,
            self.body_crc32,
            self.body_len,
            self.block_key_hash,
            self.model_id_hash,
            self.tp_rank,
            self.kv_group_id,
            self.cache_layout,
            self.dtype_code,
            self.block_size_tokens,
            self.page_size_bytes,
            self.layer_count,
            self.num_kv_heads,
            self.head_size,
            self.tensor_format,
            self.canonical_order,
            self.created_at_unix_ms,
            self.reserved,
        )

    def with_payload(self, payload: bytes) -> "KvblkHeader":
        body_len = len(payload)
        body_crc32 = _crc32(payload)
        header = replace(
            self,
            header_len=HEADER_LEN_V1,
            body_len=body_len,
            body_crc32=body_crc32,
            header_crc32=0,
        )
        header_crc32 = _crc32(header._pack())
        return replace(header, header_crc32=header_crc32)

    def to_bytes(self) -> bytes:
        return self._pack()

    @classmethod
    def from_bytes(cls, raw_header: bytes) -> "KvblkHeader":
        if len(raw_header) < HEADER_LEN_V1:
            raise KvblkFormatError("truncated header")

        unpacked = _HEADER_STRUCT.unpack(raw_header[:HEADER_LEN_V1])
        magic = unpacked[0]
        if magic != MAGIC_V1:
            raise KvblkFormatError("invalid magic")

        header = cls(
            version=unpacked[1],
            header_len=unpacked[2],
            flags=unpacked[3],
            checksum_type=unpacked[4],
            payload_codec=unpacked[5],
            header_crc32=unpacked[6],
            body_crc32=unpacked[7],
            body_len=unpacked[8],
            block_key_hash=unpacked[9],
            model_id_hash=unpacked[10],
            tp_rank=unpacked[11],
            kv_group_id=unpacked[12],
            cache_layout=unpacked[13],
            dtype_code=unpacked[14],
            block_size_tokens=unpacked[15],
            page_size_bytes=unpacked[16],
            layer_count=unpacked[17],
            num_kv_heads=unpacked[18],
            head_size=unpacked[19],
            tensor_format=unpacked[20],
            canonical_order=unpacked[21],
            created_at_unix_ms=unpacked[22],
            reserved=unpacked[23],
        )

        expected_crc = _crc32(replace(header, header_crc32=0)._pack())
        if expected_crc != header.header_crc32:
            raise KvblkFormatError("invalid header crc")

        return header


def serialize_kvblk(header: KvblkHeader, payload: bytes) -> bytes:
    """Serialize a kvblk payload using the fixed V1 header."""

    normalized_header = header.with_payload(payload)
    return normalized_header.to_bytes() + payload


def deserialize_kvblk(data: bytes) -> tuple[KvblkHeader, bytes]:
    """Deserialize kvblk bytes into a validated header and raw payload."""

    if len(data) < HEADER_LEN_V1:
        raise KvblkFormatError("truncated header")

    header = KvblkHeader.from_bytes(data[:HEADER_LEN_V1])
    payload_end = HEADER_LEN_V1 + header.body_len
    if len(data) < payload_end:
        raise KvblkFormatError("truncated body")

    payload = data[HEADER_LEN_V1:payload_end]
    if _crc32(payload) != header.body_crc32:
        raise KvblkFormatError("invalid body crc")

    return header, payload
