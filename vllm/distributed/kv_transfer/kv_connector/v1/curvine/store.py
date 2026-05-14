# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

KVBLK_FILE_SUFFIX = ".kvblk"


class CurvineStoreError(RuntimeError):
    """Base error for Curvine store operations."""


class BlockNotFoundError(CurvineStoreError):
    """Raised when a block does not exist in the store."""


@dataclass(frozen=True, slots=True)
class CurvineStoreIdentity:
    """Stable layout metadata shared by Curvine store backends (kvblk header)."""

    model_id: str
    tp_rank: int
    kv_group_id: int


class CurvineStoreClient(ABC):
    """Stable byte-oriented block store interface for Curvine backends."""

    @property
    @abstractmethod
    def store_identity(self) -> CurvineStoreIdentity:
        raise NotImplementedError

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

    @property
    def store_identity(self) -> CurvineStoreIdentity:
        return CurvineStoreIdentity(
            model_id=self._model_id,
            tp_rank=self._tp_rank,
            kv_group_id=self._kv_group_id,
        )

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


class NativeCurvineStoreClient(CurvineStoreClient):
    """Block store on Curvine FS via the Python SDK (no FUSE mount)."""

    def __init__(
        self,
        config_path: str,
        root_prefix: str,
        model_id: str,
        tp_rank: int,
        kv_group_id: int,
        *,
        write_chunk_num: int = 8,
        write_chunk_size: int = 128 * 1024 * 1024,
        cv_client: Any | None = None,
    ) -> None:
        self._model_id_sanitized = _sanitize_segment(model_id)
        self._tp_rank = tp_rank
        self._kv_group_id = kv_group_id
        rp = (root_prefix or "/").strip()
        if not rp.startswith("/"):
            rp = "/" + rp
        self._root_prefix = rp.rstrip("/") or "/"

        if cv_client is not None:
            self._cv = cv_client
        else:
            try:
                from curvinefs.curvineClient import CurvineClient
            except ImportError as e:
                raise ImportError(
                    "curvine_backend=native requires the curvine_libsdk wheel "
                    "with the curvinefs package installed"
                ) from e
            self._cv = CurvineClient(config_path, write_chunk_num, write_chunk_size)

    @property
    def store_identity(self) -> CurvineStoreIdentity:
        return CurvineStoreIdentity(
            model_id=self._model_id_sanitized,
            tp_rank=self._tp_rank,
            kv_group_id=self._kv_group_id,
        )

    def _block_path_str(self, block_key: str) -> str:
        safe_block_key = _sanitize_segment(block_key)
        hash_prefix = hashlib.sha256(block_key.encode("utf-8")).hexdigest()[:2]
        rel = "/".join(
            (
                self._model_id_sanitized,
                f"tp-{self._tp_rank}",
                f"kg-{self._kv_group_id}",
                hash_prefix,
                f"{safe_block_key}{KVBLK_FILE_SUFFIX}",
            )
        )
        if self._root_prefix == "/":
            return f"/{rel}"
        return f"{self._root_prefix}/{rel}"

    def build_block_path(self, block_key: str) -> PurePosixPath:
        return PurePosixPath(self._block_path_str(block_key))

    def exists(self, block_key: str) -> bool:
        return self._cv.path_exists(self._block_path_str(block_key))

    def batch_exists(self, block_keys: list[str]) -> list[bool]:
        return [self.exists(key) for key in block_keys]

    def read_block(self, block_key: str) -> bytes:
        path = self._block_path_str(block_key)
        if not self._cv.path_exists(path):
            raise BlockNotFoundError(f"Block not found: {block_key}")
        return self._cv.read_range(path, 0, -1)

    def batch_read(self, block_keys: list[str]) -> list[bytes | Exception]:
        items: list[bytes | Exception] = []
        for block_key in block_keys:
            try:
                items.append(self.read_block(block_key))
            except Exception as exc:  # noqa: BLE001
                items.append(exc)
        return items

    def write_block(self, block_key: str, payload: bytes) -> None:
        final_path = self._block_path_str(block_key)
        parent = str(PurePosixPath(final_path).parent)
        tmp_path = final_path + ".tmp"
        self._cv.mkdir(parent, True)
        try:
            if self._cv.path_exists(tmp_path):
                self._cv.rm(tmp_path, recursive=False)
        except OSError:
            pass
        writer = self._cv.create(tmp_path, True)
        try:
            writer.write(payload)
        finally:
            writer.close()
        try:
            if self._cv.path_exists(final_path):
                self._cv.rm(final_path, recursive=False)
        except OSError:
            pass
        self._cv.rename(tmp_path, final_path)

    def batch_write(self, items: list[tuple[str, bytes]]) -> None:
        for block_key, payload in items:
            self.write_block(block_key, payload)

    def delete_block(self, block_key: str) -> None:
        path = self._block_path_str(block_key)
        if not self._cv.path_exists(path):
            return
        self._cv.rm(path, recursive=False)


def make_curvine_store_client(
    extra_config: Mapping[str, Any] | None,
    *,
    default_model_id: str,
) -> CurvineStoreClient:
    """Factory for POSIX (FUSE layout) vs native Curvine Python SDK backends."""
    cfg = dict(extra_config or {})
    backend = str(cfg.get("curvine_backend", "posix")).lower().strip()
    model_id = cfg.get("curvine_model_id") or default_model_id
    tp_rank = int(cfg.get("curvine_tp_rank", 0))
    kv_group_id = int(cfg.get("curvine_kv_group_id", 0))

    if backend == "posix":
        root = cfg.get("curvine_store_root") or cfg.get("shared_storage_path") or "/mnt/curvine"
        return PosixCurvineStoreClient(root, model_id, tp_rank, kv_group_id)

    if backend == "native":
        config_path = cfg.get("curvine_sdk_config_path")
        if not config_path:
            raise ValueError(
                "curvine_backend=native requires curvine_sdk_config_path in kv_connector_extra_config"
            )
        root_prefix = cfg.get("curvine_native_root", "/curvine_kv")
        wn = int(cfg.get("curvine_write_chunk_num", 8))
        ws = int(cfg.get("curvine_write_chunk_size", 128 * 1024 * 1024))
        return NativeCurvineStoreClient(
            config_path=str(config_path),
            root_prefix=str(root_prefix),
            model_id=model_id,
            tp_rank=tp_rank,
            kv_group_id=kv_group_id,
            write_chunk_num=wn,
            write_chunk_size=ws,
        )

    raise ValueError(f"Unknown curvine_backend: {backend!r}; expected 'posix' or 'native'")
