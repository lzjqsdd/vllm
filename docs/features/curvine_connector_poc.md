# Curvine Connector CPU PoC

This document summarizes the current Curvine integration work for vLLM external KV storage. It is intended for contributors who need the design context, current implementation status, verification scope, and next steps in one place.

!!! note
    This work is still a PoC. The goal is to validate the connector shape, storage abstraction, and block serialization format before moving to a production-grade connector.

## Background

The current direction is to adapt Curvine as an external L2 KV store for vLLM.

The agreed implementation path is:

1. Build a PoC first, instead of starting from `OffloadingFirst`.
2. Use `KVConnectorBase_V1` as the main integration point.
3. Use Curvine FUSE plus POSIX file I/O first, then preserve a clean abstraction boundary so the backend can later switch to a native Curvine client.
4. Use a stable block file format (`kvblk`) from the beginning, so storage backend changes do not affect connector semantics.

The closest reference in vLLM is `HF3FSKVConnector`, because it already follows the external file-backed KV connector pattern.

## Design Goals

The PoC only tries to prove three things:

- vLLM KV blocks can be serialized into a stable external object format.
- The scheduler and worker connector lifecycle can drive load and save through Curvine-backed storage.
- The same connector flow can survive a future storage backend swap from FUSE/POSIX to a native Curvine client.

The PoC explicitly does not try to solve everything at once:

- No production metadata service.
- No complex manifest layer.
- No async transfer pipeline beyond the minimum connector lifecycle.
- No real GPU gather and scatter path yet.
- No multi-rank or multi-node correctness guarantees yet.

## Current Architecture

The implementation is intentionally split into three layers:

### `CurvineKVConnector`

This is the vLLM-facing integration layer.

Main responsibilities:

- Implement `KVConnectorBase_V1`.
- Build load and save plans from scheduler-side block state.
- Trigger worker-side save and load operations.
- Inject loaded payloads back into registered KV caches.

Current implementation lives in:

- `vllm/distributed/kv_transfer/kv_connector/v1/curvine/connector.py`

### `kvblk`

This is the stable block serialization format used by the PoC.

Main responsibilities:

- Encode one logical KV block into one binary object.
- Preserve enough header metadata for validation and future compatibility.
- Decode bytes back into block payloads without depending on the storage backend.

Current implementation lives in:

- `vllm/distributed/kv_transfer/kv_connector/v1/curvine/kvblk.py`

### `CurvineStoreClient`

This is the storage abstraction.

Main responsibilities:

- Map `block_key` to a storage object path.
- Support existence checks, reads, writes, and deletes.
- Hide whether the underlying backend is POSIX/FUSE or a future native Curvine client.

Current implementation lives in:

- `vllm/distributed/kv_transfer/kv_connector/v1/curvine/store.py`

The current PoC backend is `PosixCurvineStoreClient`.

## `kvblk` V1 Format

The current PoC stores one KV block per file and uses a fixed-size header plus raw payload bytes.

Important V1 properties:

- One file per block.
- Fixed header length for fast validation.
- Raw payload bytes for now.
- CRC-based validation for header and body.
- Layout and dtype metadata stored alongside the payload.

The format is designed so that:

- `PosixCurvineStoreClient` and a future native Curvine backend can share the exact same encoded bytes.
- Validation failures become explicit load errors instead of silent corruption.

## Current Object Layout

The current path strategy is:

```text
<root>/<model_id>/<tp_rank>/<kv_group>/<hash_prefix>/<block_key>.kvblk
```

This keeps object identity stable across storage backends while reducing directory hot spots.

## Current PoC Scope

The current PoC is CPU-first and layer-aware.

What is already in scope:

- Scheduler-side block hit detection.
- Scheduler metadata generation for load and save.
- Worker-side block save through `save_kv_layer`.
- Worker-side block load through `start_load_kv` and `wait_for_layer_load`.
- CPU tensor serialization and deserialization.
- Layer-scoped storage keys.
- Slot-mapping-aware load-side partial token scatter.

What is intentionally not done yet:

- Save-side real partial-token gather from scattered slots.
- Real GPU gather and scatter.
- Full async save completion semantics in `wait_for_save()` and `get_finished()`.
- Multi-rank and multi-card validation.
- Real Curvine FUSE stress validation under many small files.

## Implementation Status

The following parts are already implemented in `vllm`:

- `CurvineKVConnector` is registered in the connector factory.
- `CurvineRequestMetadata` and `CurvineConnectorMetadata` carry per-request load and save plans.
- `save_kv_layer()` can persist raw payloads and CPU tensors as `kvblk`.
- `start_load_kv()` builds per-layer pending load queues.
- `wait_for_layer_load(layer_name)` performs deferred per-layer injection.
- Store keys are layer-scoped, preventing collisions between layers for the same logical block key.
- Corrupted `kvblk` objects are treated as load failures.

The CPU path has already progressed beyond simple bytes round-trips:

- Full block extraction from CPU KV tensors is implemented.
- Full block reinjection into registered CPU KV caches is implemented.
- `ForwardContext.slot_mapping[layer_name]` is consumed on the load path.
- If only some tokens of a block are requested, the current PoC only scatters those tokens back instead of overwriting the entire block.
- If save-side `slot_mapping` only covers part of a block, the current PoC skips persistence to avoid writing partial or dirty blocks.

## Current Tests

The current Curvine PoC work is covered by focused unit tests:

- `tests/v1/kv_connector/unit/test_curvine_kvblk.py`
- `tests/v1/kv_connector/unit/test_curvine_store.py`
- `tests/v1/kv_connector/unit/test_curvine_connector.py`

These tests currently cover:

- `kvblk` header and encode/decode behavior.
- Store path mapping and POSIX read/write behavior.
- Connector factory registration.
- Scheduler-side matched block counting.
- Scheduler-side load and save metadata generation.
- Worker-side save and load through metadata.
- CPU tensor block extraction and reinjection.
- Corrupted block handling.
- Layer-scoped save and load behavior.
- Load-side partial token scatter driven by `slot_mapping`.

## Environment Notes

For lightweight Curvine connector development, the repository already contains:

- `requirements/curvine_connector_poc.txt`

This file includes:

```text
-r common.txt
-r kv_connectors.txt
pytest
```

It is suitable for focused connector unit testing.

## Current CPU End-to-End Status

There are two distinct CPU validation layers:

### 1. Connector-focused CPU tests

This layer is already working and is the main source of confidence for the PoC today.

### 2. Full `LLM.generate()` CPU end-to-end validation

This is not fully closed yet in the current workspace.

The latest environment investigation found:

- A precompiled CPU-flavored editable install can start the CPU engine, load the model, and reach execution.
- However, it is still missing compiled custom ops needed by the runtime path, such as `torch.ops._C.compute_slot_mapping_kernel_impl`.
- Switching to a full source CPU build also requires the Python environment to use a CPU build of PyTorch and a non-isolated build path that compiles vLLM custom CPU extensions successfully.

This means the remaining CPU end-to-end gap is now mainly an environment and compiled-extension issue, not a gap in the Curvine connector Python logic itself.

## Risks

Current high-risk areas are:

- FUSE latency under many small KV block files.
- Mismatch between serialized canonical layout and runtime KV layout.
- Incorrect `block_key` semantics causing false hits or missed hits.
- Metadata pressure from one-file-per-block storage.
- Save-side partial block handling still being conservative.

Current mitigations are:

- `kvblk` validation on load.
- Clean `CurvineStoreClient` boundary.
- CPU-first narrowing of runtime semantics before GPU integration.
- Focused tests around layer-scoped and slot-aware behavior.

## Next Steps

The current execution order should be:

1. Finish save-side partial-token gather on CPU.
2. Close the full CPU `LLM.generate()` end-to-end path by fixing the CPU build and custom op environment.
3. Run a true Curvine-backed CPU PoC with real requests.
4. Port the worker path from CPU-only semantics to real GPU gather and scatter.
5. Validate multi-rank and stress scenarios.

## Quick Contributor Map

If you need to continue this work, start from these files:

- `vllm/distributed/kv_transfer/kv_connector/v1/curvine/connector.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/curvine/kvblk.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/curvine/store.py`
- `tests/v1/kv_connector/unit/test_curvine_connector.py`
- `tests/v1/kv_connector/unit/test_curvine_kvblk.py`
- `tests/v1/kv_connector/unit/test_curvine_store.py`
- `requirements/curvine_connector_poc.txt`

For the original broader design context that led to this PoC, refer to the Curvine-side design notes and then sync any contributor-facing updates back into this document.
