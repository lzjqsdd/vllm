# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

KVBLK_FILE_SUFFIX = ".kvblk"


class CurvineStoreError(RuntimeError):
    """Base error for Curvine store operations."""


class BlockNotFoundError(CurvineStoreError):
    """Raised when a block does not exist in the store."""


class CurvineStoreClient(ABC):
    """Stable byte-oriented block store interface for Curvine backends."""

    @abstractmethod
    def exists(self, block_key: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def batch_exists(self, block_keys: list[str]) -> list[bool]:
        raise NotImplementedError

    @abstractmethod
    def read_block(self, block_key: str) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def batch_read(self, block_keys: list[str]) -> list[bytes | Exception]:
        raise NotImplementedError

    @abstractmethod
    def write_block(self, block_key: str, payload: bytes) -> None:
        raise NotImplementedError

    @abstractmethod
    def batch_write(self, items: list[tuple[str, bytes]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_block(self, block_key: str) -> None:
        raise NotImplementedError


def _sanitize_segment(value: str) -> str:
    sanitized = value.replace(os.sep, "_")
    if os.altsep:
        sanitized = sanitized.replace(os.altsep, "_")
    return sanitized


class PosixCurvineStoreClient(CurvineStoreClient):
    """PoC store backend using a CurvineFuse-mounted POSIX directory."""

    def __init__(
        self,
        root_dir: str | Path,
        model_id: str,
        tp_rank: int,
        kv_group_id: int,
    ) -> None:
        self._root_dir = Path(root_dir)
        self._model_id = _sanitize_segment(model_id)
        self._tp_rank = tp_rank
        self._kv_group_id = kv_group_id

    def build_block_path(self, block_key: str) -> Path:
        safe_block_key = _sanitize_segment(block_key)
        hash_prefix = hashlib.sha256(block_key.encode("utf-8")).hexdigest()[:2]
        return (
            self._root_dir
            / self._model_id
            / f"tp-{self._tp_rank}"
            / f"kg-{self._kv_group_id}"
            / hash_prefix
            / f"{safe_block_key}{KVBLK_FILE_SUFFIX}"
        )

    def exists(self, block_key: str) -> bool:
        return self.build_block_path(block_key).exists()

    def batch_exists(self, block_keys: list[str]) -> list[bool]:
        return [self.exists(block_key) for block_key in block_keys]

    def read_block(self, block_key: str) -> bytes:
        block_path = self.build_block_path(block_key)
        if not block_path.exists():
            raise BlockNotFoundError(f"Block not found: {block_key}")
        return block_path.read_bytes()

    def batch_read(self, block_keys: list[str]) -> list[bytes | Exception]:
        items: list[bytes | Exception] = []
        for block_key in block_keys:
            try:
                items.append(self.read_block(block_key))
            except Exception as exc:  # noqa: BLE001
                items.append(exc)
        return items

    def write_block(self, block_key: str, payload: bytes) -> None:
        block_path = self.build_block_path(block_key)
        block_path.parent.mkdir(parents=True, exist_ok=True)

        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=block_path.parent,
                prefix=f"{block_path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = handle.name
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temp_path, block_path)
        finally:
            if temp_path is not None and os.path.exists(temp_path):
                os.remove(temp_path)

    def batch_write(self, items: list[tuple[str, bytes]]) -> None:
        for block_key, payload in items:
            self.write_block(block_key, payload)

    def delete_block(self, block_key: str) -> None:
        block_path = self.build_block_path(block_key)
        if block_path.exists():
            block_path.unlink()
