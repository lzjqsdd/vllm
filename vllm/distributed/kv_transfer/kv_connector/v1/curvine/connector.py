# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.curvine.kvblk import (
    CacheLayout,
    CanonicalOrder,
    ChecksumType,
    DTypeCode,
    KvblkHeader,
    KvblkFormatError,
    PayloadCodec,
    TensorFormat,
    deserialize_kvblk,
    serialize_kvblk,
)
from vllm.distributed.kv_transfer.kv_connector.v1.curvine.store import (
    BlockNotFoundError,
    PosixCurvineStoreClient,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass(slots=True)
class CurvineRequestMetadata:
    request_id: str
    operation: Literal["load", "save"]
    block_ids: list[int]
    block_keys: list[str]
    token_count: int


class CurvineConnectorMetadata(KVConnectorMetadata):
    def __init__(self) -> None:
        self.requests: list[CurvineRequestMetadata] = []

    def add_request(self, request_metadata: CurvineRequestMetadata) -> None:
        self.requests.append(request_metadata)


@dataclass(slots=True)
class CurvineSchedulingState:
    request_id: str
    request: "Request | None" = None
    num_saved_blocks: int = 0
    load_block_keys: list[str] = field(default_factory=list)
    allocated_block_ids: list[int] = field(default_factory=list)
    phase: str = "NEW"

    def needs_loading(self) -> bool:
        return bool(self.load_block_keys)

    def is_ready_to_load(self) -> bool:
        return self.phase == "WAITING_TO_LOAD" and self.needs_loading()


class CurvineKVConnector(KVConnectorBase_V1):
    """Minimal PoC connector for Curvine-backed KV block storage."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        extra_config = self._kv_transfer_config.kv_connector_extra_config
        self._block_size = vllm_config.cache_config.block_size
        self._store = PosixCurvineStoreClient(
            root_dir=extra_config.get("curvine_store_root")
            or extra_config.get("shared_storage_path")
            or "/tmp/curvine-kv",
            model_id=extra_config.get("curvine_model_id")
            or vllm_config.model_config.model,
            tp_rank=int(extra_config.get("curvine_tp_rank", 0)),
            kv_group_id=int(extra_config.get("curvine_kv_group_id", 0)),
        )
        self._scheduling_states: dict[str, CurvineSchedulingState] = {}
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._loaded_payloads_by_req: dict[str, dict[str, bytes]] = {}
        self._failed_load_block_ids: set[int] = set()
        self._pending_layer_loads: dict[
            str, list[tuple[str, int, str, torch.Tensor | None]]
        ] = {}

    @property
    def store_client(self) -> PosixCurvineStoreClient:
        return self._store

    def block_key_for_hash(self, block_hash: bytes) -> str:
        return bytes(block_hash).hex()

    def block_keys_from_hashes(self, block_hashes: list[bytes]) -> list[str]:
        return [self.block_key_for_hash(block_hash) for block_hash in block_hashes]

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._kv_caches = kv_caches

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, CurvineConnectorMetadata)
        self._failed_load_block_ids.clear()
        self._pending_layer_loads.clear()

        for request_metadata in metadata.requests:
            if request_metadata.operation != "load":
                continue

            self._loaded_payloads_by_req[request_metadata.request_id] = {}
            for block_id, block_key in zip(
                request_metadata.block_ids,
                request_metadata.block_keys,
            ):
                if self._kv_caches:
                    for layer_name in self._kv_caches:
                        layer_slot_mapping = self._resolve_forward_slot_mapping(
                            forward_context, layer_name
                        )
                        self._pending_layer_loads.setdefault(layer_name, []).append(
                            (
                                request_metadata.request_id,
                                block_id,
                                block_key,
                                layer_slot_mapping,
                            )
                        )
                else:
                    self._pending_layer_loads.setdefault("", []).append(
                        (request_metadata.request_id, block_id, block_key, None)
                    )

        if not self._kv_caches:
            self._consume_pending_layer_loads(layer_name="", kv_cache=None)

    def wait_for_layer_load(self, layer_name: str) -> None:
        target_layer_name = layer_name if layer_name in self._kv_caches else ""
        kv_cache = self._kv_caches.get(layer_name)
        if kv_cache is None:
            kv_cache = self._get_single_target_kv_cache()
        self._consume_pending_layer_loads(layer_name=target_layer_name, kv_cache=kv_cache)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata,
        **kwargs: Any,
    ) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, CurvineConnectorMetadata)
        block_payloads_by_key = kwargs.get("block_payloads_by_key", {})
        block_tensors_by_key = kwargs.get("block_tensors_by_key", {})

        for request_metadata in metadata.requests:
            if request_metadata.operation != "save":
                continue

            items = []
            for block_id, block_key in zip(
                request_metadata.block_ids,
                request_metadata.block_keys,
            ):
                store_block_key = self._resolve_save_block_key(block_key, layer_name)
                encoded = self._encode_block_from_kwargs(
                    block_key=store_block_key,
                    block_id=block_id,
                    kv_layer=kv_layer,
                    attn_metadata=attn_metadata,
                    block_payloads_by_key=block_payloads_by_key,
                    block_tensors_by_key=block_tensors_by_key,
                )
                if encoded is not None:
                    items.append((store_block_key, encoded))
            if items:
                self._store.batch_write(items)

    def wait_for_save(self):
        return

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        return None, None

    def get_block_ids_with_load_errors(self) -> set[int]:
        return set(self._failed_load_block_ids)

    def get_loaded_block_payloads(self, request_id: str) -> dict[str, bytes]:
        return self._loaded_payloads_by_req.get(request_id, {})

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        state = self._get_or_create_state(request.request_id)
        state.request = request

        num_full_blocks = self._num_full_blocks(len(request.prompt_token_ids or []))
        if num_full_blocks <= 0:
            state.load_block_keys = []
            return 0, False

        start_block = num_computed_tokens // self._block_size
        request_block_keys = self.block_keys_from_hashes(request.block_hashes[:num_full_blocks])
        exists_results = self._store.batch_exists(request_block_keys)
        matched_blocks = next(
            (index for index, exists in enumerate(exists_results) if not exists),
            len(exists_results),
        )
        if matched_blocks <= start_block:
            state.load_block_keys = []
            return 0, False

        state.load_block_keys = request_block_keys[start_block:matched_blocks]
        return len(state.load_block_keys) * self._block_size, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        if num_external_tokens <= 0:
            return

        state = self._get_or_create_state(request.request_id)
        allocated_block_ids = list(blocks.get_unhashed_block_ids())
        expected_blocks = num_external_tokens // self._block_size
        state.allocated_block_ids = allocated_block_ids[:expected_blocks]
        state.phase = "WAITING_TO_LOAD"

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        metadata = CurvineConnectorMetadata()

        for request_id in scheduler_output.finished_req_ids:
            self._scheduling_states.pop(request_id, None)

        self._append_load_requests(metadata)
        self._append_new_request_saves(scheduler_output, metadata)
        self._append_cached_request_saves(scheduler_output, metadata)
        return metadata

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        self._scheduling_states.pop(request.request_id, None)
        return False, None

    def _append_load_requests(self, metadata: CurvineConnectorMetadata) -> None:
        for state in self._scheduling_states.values():
            if not state.is_ready_to_load():
                continue

            metadata.add_request(
                CurvineRequestMetadata(
                    request_id=state.request_id,
                    operation="load",
                    block_ids=state.allocated_block_ids.copy(),
                    block_keys=state.load_block_keys.copy(),
                    token_count=len(state.load_block_keys) * self._block_size,
                )
            )
            state.phase = "ACTIVE"
            state.load_block_keys = []
            state.allocated_block_ids = []

    def _append_new_request_saves(
        self,
        scheduler_output: SchedulerOutput,
        metadata: CurvineConnectorMetadata,
    ) -> None:
        for request in scheduler_output.scheduled_new_reqs:
            state = self._get_or_create_state(request.req_id)
            if state.request is None:
                continue

            total_tokens = request.num_computed_tokens + scheduler_output.num_scheduled_tokens.get(
                request.req_id,
                0,
            )
            self._append_save_request(
                metadata=metadata,
                state=state,
                block_ids=self._normalize_block_ids(request.block_ids),
                total_tokens=total_tokens,
            )

    def _append_cached_request_saves(
        self,
        scheduler_output: SchedulerOutput,
        metadata: CurvineConnectorMetadata,
    ) -> None:
        cached_reqs: CachedRequestData = scheduler_output.scheduled_cached_reqs
        for index, request_id in enumerate(cached_reqs.req_ids):
            state = self._scheduling_states.get(request_id)
            if state is None or state.request is None:
                continue

            total_tokens = cached_reqs.num_computed_tokens[index] + scheduler_output.num_scheduled_tokens.get(
                request_id,
                0,
            )
            block_ids = self._normalize_block_ids(cached_reqs.new_block_ids[index])
            if not block_ids:
                continue

            self._append_save_request(
                metadata=metadata,
                state=state,
                block_ids=block_ids,
                total_tokens=total_tokens,
            )

    def _append_save_request(
        self,
        metadata: CurvineConnectorMetadata,
        state: CurvineSchedulingState,
        block_ids: list[int],
        total_tokens: int,
    ) -> None:
        total_full_blocks = self._num_full_blocks(total_tokens)
        if total_full_blocks <= state.num_saved_blocks:
            return

        block_keys = self.block_keys_from_hashes(
            state.request.block_hashes[state.num_saved_blocks:total_full_blocks]
        )
        num_new_blocks = len(block_keys)
        if num_new_blocks <= 0:
            return

        metadata.add_request(
            CurvineRequestMetadata(
                request_id=state.request_id,
                operation="save",
                block_ids=block_ids[:num_new_blocks],
                block_keys=block_keys,
                token_count=total_full_blocks * self._block_size,
            )
        )
        state.num_saved_blocks = total_full_blocks

    def _get_or_create_state(self, request_id: str) -> CurvineSchedulingState:
        if request_id not in self._scheduling_states:
            self._scheduling_states[request_id] = CurvineSchedulingState(
                request_id=request_id
            )
        return self._scheduling_states[request_id]

    def _normalize_block_ids(self, block_ids: Any) -> list[int]:
        if block_ids is None:
            return []
        if isinstance(block_ids, tuple):
            return list(block_ids[0]) if block_ids else []
        return list(block_ids)

    def _num_full_blocks(self, token_count: int) -> int:
        return token_count // self._block_size

    def _encode_block_from_kwargs(
        self,
        block_key: str,
        block_id: int,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        block_payloads_by_key: Any,
        block_tensors_by_key: Any,
    ) -> bytes | None:
        if isinstance(block_payloads_by_key, dict) and block_key in block_payloads_by_key:
            payload = block_payloads_by_key[block_key]
            if isinstance(payload, bytes):
                return self._serialize_block_bytes(
                    block_key=block_key,
                    payload=payload,
                    dtype_code=DTypeCode.INT8,
                )

        if isinstance(block_tensors_by_key, dict) and block_key in block_tensors_by_key:
            tensor = block_tensors_by_key[block_key]
            if isinstance(tensor, torch.Tensor):
                payload = self._tensor_to_bytes(tensor)
                return self._serialize_block_bytes(
                    block_key=block_key,
                    payload=payload,
                    dtype_code=self._dtype_code_for_tensor(tensor),
                )

        if isinstance(kv_layer, torch.Tensor) and kv_layer.numel() > 0:
            slot_mapping = getattr(attn_metadata, "slot_mapping", None)
            if isinstance(slot_mapping, torch.Tensor):
                block_tensor_from_slots = self._extract_block_tensor_from_slot_mapping(
                    kv_layer=kv_layer,
                    block_id=block_id,
                    slot_mapping=slot_mapping,
                )
                if block_tensor_from_slots is not None:
                    block_tensor, cache_layout = block_tensor_from_slots
                    return self._serialize_block_tensor(
                        block_key=block_key,
                        block_tensor=block_tensor,
                        cache_layout=cache_layout,
                    )
                if self._slot_mapping_mentions_block(slot_mapping, block_id):
                    return None

            block_tensor, cache_layout = self._extract_block_tensor(kv_layer, block_id)
            return self._serialize_block_tensor(
                block_key=block_key,
                block_tensor=block_tensor,
                cache_layout=cache_layout,
            )

        return None

    def _serialize_block_bytes(
        self,
        block_key: str,
        payload: bytes,
        dtype_code: int,
        *,
        cache_layout: int = CacheLayout.NHD,
        num_kv_heads: int = 0,
        head_size: int = 0,
    ) -> bytes:
        header = KvblkHeader(
            checksum_type=ChecksumType.CRC32,
            payload_codec=PayloadCodec.RAW,
            block_key_hash=hashlib.md5(block_key.encode("utf-8")).digest(),
            model_id_hash=self._model_id_hash(),
            tp_rank=self._store._tp_rank,
            kv_group_id=self._store._kv_group_id,
            cache_layout=cache_layout,
            dtype_code=dtype_code,
            block_size_tokens=self._block_size,
            page_size_bytes=len(payload),
            layer_count=1,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            tensor_format=TensorFormat.FULL_KV_SEPARATE,
            canonical_order=CanonicalOrder.LAYER_MAJOR_K_THEN_V,
            created_at_unix_ms=int(time.time() * 1000),
        )
        return serialize_kvblk(header, payload)

    def _serialize_block_tensor(
        self,
        block_key: str,
        block_tensor: torch.Tensor,
        cache_layout: int,
    ) -> bytes:
        payload = self._tensor_to_bytes(block_tensor)
        if cache_layout == CacheLayout.NHD:
            _, _, num_kv_heads, head_size = block_tensor.shape
        else:
            _, num_kv_heads, _, head_size = block_tensor.shape
        return self._serialize_block_bytes(
            block_key=block_key,
            payload=payload,
            dtype_code=self._dtype_code_for_tensor(block_tensor),
            cache_layout=cache_layout,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )

    def _model_id_hash(self) -> int:
        model_digest = hashlib.sha256(self._store._model_id.encode("utf-8")).digest()
        return int.from_bytes(model_digest[:8], "little", signed=False)

    def _tensor_to_bytes(self, tensor: torch.Tensor) -> bytes:
        cpu_tensor = tensor.detach().cpu().contiguous()
        if cpu_tensor.dtype == torch.bfloat16:
            return cpu_tensor.view(dtype=torch.uint16).numpy().tobytes()
        return cpu_tensor.numpy().tobytes()

    def _dtype_code_for_tensor(self, tensor: torch.Tensor) -> int:
        dtype_map = {
            torch.float16: DTypeCode.FP16,
            torch.bfloat16: DTypeCode.BF16,
            torch.float32: DTypeCode.FP32,
            torch.int8: DTypeCode.INT8,
        }
        return dtype_map.get(tensor.dtype, DTypeCode.INT8)

    def _extract_block_tensor(
        self,
        kv_layer: torch.Tensor,
        block_id: int,
    ) -> tuple[torch.Tensor, int]:
        if kv_layer.ndim != 5:
            raise ValueError(f"Unsupported kv_layer ndim: {kv_layer.ndim}")

        if kv_layer.shape[1] == 2 and kv_layer.shape[2] == self._block_size:
            return kv_layer[block_id].detach().cpu().contiguous(), CacheLayout.NHD

        if kv_layer.shape[1] == 2 and kv_layer.shape[3] == self._block_size:
            return kv_layer[block_id].detach().cpu().contiguous(), CacheLayout.HND

        if kv_layer.shape[0] == 2 and kv_layer.shape[3] == self._block_size:
            return kv_layer[:, block_id].detach().cpu().contiguous(), CacheLayout.HND

        raise ValueError(f"Unsupported kv_layer shape: {tuple(kv_layer.shape)}")

    def _extract_block_tensor_from_slot_mapping(
        self,
        kv_layer: torch.Tensor,
        block_id: int,
        slot_mapping: torch.Tensor,
    ) -> tuple[torch.Tensor, int] | None:
        block_slots = self._get_full_block_slots(slot_mapping, block_id)
        if block_slots is None:
            return None

        if kv_layer.ndim != 5:
            raise ValueError(f"Unsupported kv_layer ndim: {kv_layer.ndim}")

        if kv_layer.shape[1] == 2 and kv_layer.shape[2] == self._block_size:
            block_tensor = torch.empty_like(kv_layer[block_id]).detach().cpu()
            for offset, slot_id in enumerate(block_slots):
                block_tensor[:, offset].copy_(kv_layer[slot_id // self._block_size, :, slot_id % self._block_size].detach().cpu())
            return block_tensor.contiguous(), CacheLayout.NHD

        if kv_layer.shape[1] == 2 and kv_layer.shape[3] == self._block_size:
            block_tensor = torch.empty_like(kv_layer[block_id]).detach().cpu()
            for offset, slot_id in enumerate(block_slots):
                block_tensor[:, :, offset].copy_(
                    kv_layer[
                        slot_id // self._block_size,
                        :,
                        :,
                        slot_id % self._block_size,
                    ].detach().cpu()
                )
            return block_tensor.contiguous(), CacheLayout.HND

        if kv_layer.shape[0] == 2 and kv_layer.shape[3] == self._block_size:
            block_tensor = torch.empty_like(kv_layer[:, block_id]).detach().cpu()
            for offset, slot_id in enumerate(block_slots):
                block_tensor[:, :, offset].copy_(
                    kv_layer[
                        :,
                        slot_id // self._block_size,
                        :,
                        slot_id % self._block_size,
                    ].detach().cpu()
                )
            return block_tensor.contiguous(), CacheLayout.HND

        raise ValueError(f"Unsupported kv_layer shape: {tuple(kv_layer.shape)}")

    def _get_full_block_slots(
        self,
        slot_mapping: torch.Tensor,
        block_id: int,
    ) -> list[int] | None:
        matching_slots = {
            int(slot_id)
            for slot_id in slot_mapping.detach().cpu().tolist()
            if int(slot_id) >= 0 and int(slot_id) // self._block_size == block_id
        }
        if len(matching_slots) != self._block_size:
            return None

        ordered_slots = sorted(matching_slots, key=lambda slot_id: slot_id % self._block_size)
        expected_slots = list(
            range(block_id * self._block_size, (block_id + 1) * self._block_size)
        )
        if ordered_slots != expected_slots:
            return None
        return ordered_slots

    def _slot_mapping_mentions_block(
        self,
        slot_mapping: torch.Tensor,
        block_id: int,
    ) -> bool:
        for slot_id in slot_mapping.detach().cpu().tolist():
            slot_id = int(slot_id)
            if slot_id >= 0 and slot_id // self._block_size == block_id:
                return True
        return False

    def _get_single_target_kv_cache(self) -> torch.Tensor | None:
        if len(self._kv_caches) != 1:
            return None
        return next(iter(self._kv_caches.values()))

    def _inject_payload_into_kv_cache(
        self,
        kv_cache: torch.Tensor,
        block_id: int,
        header: KvblkHeader,
        payload: bytes,
        slot_mapping: torch.Tensor | None = None,
    ) -> None:
        block_tensor = self._payload_to_block_tensor(header, payload)

        if kv_cache.ndim != 5:
            raise ValueError(f"Unsupported kv_cache ndim: {kv_cache.ndim}")

        if isinstance(slot_mapping, torch.Tensor) and self._slot_mapping_mentions_block(
            slot_mapping, block_id
        ):
            self._scatter_block_tensor_by_slot_mapping(
                kv_cache=kv_cache,
                block_tensor=block_tensor,
                cache_layout=header.cache_layout,
                slot_mapping=slot_mapping,
                block_id=block_id,
            )
            return

        if kv_cache.shape[1] == 2 and kv_cache.shape[2] == self._block_size:
            kv_cache[block_id].copy_(self._to_nhd_tensor(block_tensor, header.cache_layout))
            return

        if kv_cache.shape[1] == 2 and kv_cache.shape[3] == self._block_size:
            kv_cache[block_id].copy_(self._to_hnd_tensor(block_tensor, header.cache_layout))
            return

        if kv_cache.shape[0] == 2 and kv_cache.shape[3] == self._block_size:
            kv_cache[:, block_id].copy_(
                self._to_hnd_tensor(block_tensor, header.cache_layout)
            )
            return

        raise ValueError(f"Unsupported kv_cache shape: {tuple(kv_cache.shape)}")

    def _payload_to_block_tensor(
        self,
        header: KvblkHeader,
        payload: bytes,
    ) -> torch.Tensor:
        dtype = self._torch_dtype_for_code(header.dtype_code)
        writable_payload = bytearray(payload)
        if dtype == torch.bfloat16:
            tensor = torch.frombuffer(writable_payload, dtype=torch.uint16).view(
                torch.bfloat16
            )
        else:
            tensor = torch.frombuffer(writable_payload, dtype=dtype)

        if header.cache_layout == CacheLayout.NHD:
            shape = (2, header.block_size_tokens, header.num_kv_heads, header.head_size)
        else:
            shape = (2, header.num_kv_heads, header.block_size_tokens, header.head_size)
        return tensor.clone().reshape(shape)

    def _to_nhd_tensor(self, block_tensor: torch.Tensor, cache_layout: int) -> torch.Tensor:
        if cache_layout == CacheLayout.NHD:
            return block_tensor
        return block_tensor.permute(0, 2, 1, 3).contiguous()

    def _to_hnd_tensor(self, block_tensor: torch.Tensor, cache_layout: int) -> torch.Tensor:
        if cache_layout == CacheLayout.HND:
            return block_tensor
        return block_tensor.permute(0, 2, 1, 3).contiguous()

    def _torch_dtype_for_code(self, dtype_code: int) -> torch.dtype:
        dtype_map = {
            DTypeCode.FP16: torch.float16,
            DTypeCode.BF16: torch.bfloat16,
            DTypeCode.FP32: torch.float32,
            DTypeCode.INT8: torch.int8,
        }
        return dtype_map.get(DTypeCode(dtype_code), torch.uint8)

    def _resolve_save_block_key(self, block_key: str, layer_name: str) -> str:
        if layer_name in self._kv_caches:
            return self._layer_scoped_block_key(block_key, layer_name)
        return block_key

    def _resolve_load_block_key(self, block_key: str, layer_name: str) -> str:
        layer_scoped_key = self._layer_scoped_block_key(block_key, layer_name)
        if layer_name in self._kv_caches and self._store.exists(layer_scoped_key):
            return layer_scoped_key
        return block_key

    def _layer_scoped_block_key(self, block_key: str, layer_name: str) -> str:
        safe_layer_name = layer_name.replace("/", "_").replace(".", "_")
        return f"{block_key}__{safe_layer_name}"

    def _resolve_forward_slot_mapping(
        self,
        forward_context: "ForwardContext | None",
        layer_name: str,
    ) -> torch.Tensor | None:
        if forward_context is None:
            return None

        slot_mapping = getattr(forward_context, "slot_mapping", None)
        if isinstance(slot_mapping, dict):
            layer_slot_mapping = slot_mapping.get(layer_name)
            if isinstance(layer_slot_mapping, torch.Tensor):
                return layer_slot_mapping
        return None

    def _consume_pending_layer_loads(
        self,
        layer_name: str,
        kv_cache: torch.Tensor | None,
    ) -> None:
        pending = self._pending_layer_loads.pop(layer_name, [])
        if not pending:
            return

        for request_id, block_id, block_key, slot_mapping in pending:
            store_block_key = self._resolve_load_block_key(block_key, layer_name)
            try:
                raw = self._store.read_block(store_block_key)
                header, payload = deserialize_kvblk(raw)
            except (BlockNotFoundError, KvblkFormatError, RuntimeError, ValueError):
                self._failed_load_block_ids.add(block_id)
                continue

            self._loaded_payloads_by_req.setdefault(request_id, {})[block_key] = payload
            if kv_cache is not None:
                self._inject_payload_into_kv_cache(
                    kv_cache=kv_cache,
                    block_id=block_id,
                    header=header,
                    payload=payload,
                    slot_mapping=slot_mapping,
                )

    def _scatter_block_tensor_by_slot_mapping(
        self,
        kv_cache: torch.Tensor,
        block_tensor: torch.Tensor,
        cache_layout: int,
        slot_mapping: torch.Tensor,
        block_id: int,
    ) -> None:
        for slot_id in slot_mapping.detach().cpu().tolist():
            slot_id = int(slot_id)
            if slot_id < 0 or slot_id // self._block_size != block_id:
                continue

            token_offset = slot_id % self._block_size
            token_slice = self._get_block_token_slice(
                block_tensor=block_tensor,
                cache_layout=cache_layout,
                token_offset=token_offset,
            )
            self._write_token_slice_into_kv_cache(
                kv_cache=kv_cache,
                block_id=block_id,
                token_offset=token_offset,
                token_slice=token_slice,
            )

    def _get_block_token_slice(
        self,
        block_tensor: torch.Tensor,
        cache_layout: int,
        token_offset: int,
    ) -> torch.Tensor:
        if cache_layout == CacheLayout.NHD:
            return block_tensor[:, token_offset].contiguous()
        return block_tensor[:, :, token_offset].contiguous()

    def _write_token_slice_into_kv_cache(
        self,
        kv_cache: torch.Tensor,
        block_id: int,
        token_offset: int,
        token_slice: torch.Tensor,
    ) -> None:
        if kv_cache.shape[1] == 2 and kv_cache.shape[2] == self._block_size:
            kv_cache[block_id, :, token_offset].copy_(token_slice)
            return

        if kv_cache.shape[1] == 2 and kv_cache.shape[3] == self._block_size:
            kv_cache[block_id, :, :, token_offset].copy_(token_slice)
            return

        if kv_cache.shape[0] == 2 and kv_cache.shape[3] == self._block_size:
            kv_cache[:, block_id, :, token_offset].copy_(token_slice)
            return

        raise ValueError(f"Unsupported kv_cache shape: {tuple(kv_cache.shape)}")
