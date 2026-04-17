#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace

import torch

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.curvine.connector import (
    CurvineConnectorMetadata,
    CurvineKVConnector,
    CurvineRequestMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.curvine.kvblk import (
    DTypeCode,
    KvblkHeader,
    deserialize_kvblk,
    serialize_kvblk,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput

from .utils import create_request, create_vllm_config


class FakeBlocks:
    def __init__(self, block_ids: list[int]):
        self._block_ids = block_ids

    def get_unhashed_block_ids(self) -> list[int]:
        return self._block_ids


def make_scheduler_output(
    request,
    block_ids: list[int],
    scheduled_tokens: int,
) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[NewRequestData.from_request(request, (block_ids,))],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request.request_id: scheduled_tokens},
        total_num_scheduled_tokens=scheduled_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


class TestCurvineKVConnector(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = create_vllm_config(
            kv_connector="CurvineKVConnector",
            kv_connector_extra_config={
                "curvine_store_root": self.temp_dir.name,
                "curvine_model_id": "unit-test-model",
                "curvine_tp_rank": 0,
                "curvine_kv_group_id": 0,
            },
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_connector(self, role: KVConnectorRole) -> CurvineKVConnector:
        return CurvineKVConnector(self.config, role)

    def make_scheduler_connector_with_layers(
        self, layer_names: list[str]
    ) -> CurvineKVConnector:
        kv_cache_config = KVCacheConfig(
            num_blocks=64,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names,
                    FullAttentionSpec(
                        block_size=16,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                )
            ],
        )
        return CurvineKVConnector(
            self.config,
            KVConnectorRole.SCHEDULER,
            kv_cache_config=kv_cache_config,
        )

    def test_factory_registers_curvine_connector(self):
        connector_cls = KVConnectorFactory.get_connector_class_by_name(
            "CurvineKVConnector"
        )
        self.assertIs(connector_cls, CurvineKVConnector)

    def test_get_num_new_matched_tokens_counts_consecutive_block_hits(self):
        connector = self.make_connector(KVConnectorRole.SCHEDULER)
        request = create_request(num_tokens=48, block_size=16)

        block_keys = connector.block_keys_from_hashes(request.block_hashes)
        connector.store_client.write_block(block_keys[0], b"block-0")
        connector.store_client.write_block(block_keys[1], b"block-1")

        matched_tokens, is_async = connector.get_num_new_matched_tokens(request, 0)

        self.assertEqual(matched_tokens, 32)
        self.assertFalse(is_async)

    def test_build_connector_meta_emits_load_request_after_alloc(self):
        connector = self.make_connector(KVConnectorRole.SCHEDULER)
        request = create_request(num_tokens=48, block_size=16)
        block_keys = connector.block_keys_from_hashes(request.block_hashes)
        connector.store_client.write_block(block_keys[0], b"block-0")
        connector.store_client.write_block(block_keys[1], b"block-1")

        matched_tokens, _ = connector.get_num_new_matched_tokens(request, 0)
        connector.update_state_after_alloc(
            request,
            FakeBlocks([101, 102]),
            matched_tokens,
        )

        metadata = connector.build_connector_meta(SchedulerOutput.make_empty())
        self.assertIsInstance(metadata, CurvineConnectorMetadata)
        self.assertEqual(len(metadata.requests), 1)

        request_metadata = metadata.requests[0]
        self.assertEqual(request_metadata.operation, "load")
        self.assertEqual(request_metadata.request_id, request.request_id)
        self.assertEqual(request_metadata.block_ids, [101, 102])
        self.assertEqual(request_metadata.block_keys, block_keys[:2])

    def test_build_connector_meta_emits_save_request_for_new_request(self):
        connector = self.make_connector(KVConnectorRole.SCHEDULER)
        request = create_request(num_tokens=48, block_size=16)

        matched_tokens, _ = connector.get_num_new_matched_tokens(request, 0)
        self.assertEqual(matched_tokens, 0)

        scheduler_output = make_scheduler_output(
            request=request,
            block_ids=[11, 12, 13],
            scheduled_tokens=48,
        )
        metadata = connector.build_connector_meta(scheduler_output)

        self.assertEqual(len(metadata.requests), 1)
        request_metadata = metadata.requests[0]
        self.assertEqual(request_metadata.operation, "save")
        self.assertEqual(request_metadata.block_ids, [11, 12, 13])
        self.assertEqual(
            request_metadata.block_keys,
            connector.block_keys_from_hashes(request.block_hashes),
        )

    def test_get_num_new_matched_tokens_recognizes_layer_scoped_block_hits(self):
        layer_names = ["model.decoder.layers.0.self_attn.attn", "model.decoder.layers.1.self_attn.attn"]
        connector = self.make_scheduler_connector_with_layers(layer_names)
        request = create_request(num_tokens=32, block_size=16)

        block_key = connector.block_keys_from_hashes(request.block_hashes)[0]
        for layer_name in layer_names:
            connector.store_client.write_block(
                connector._layer_scoped_block_key(block_key, layer_name),
                f"{layer_name}-payload".encode(),
            )

        matched_tokens, is_async = connector.get_num_new_matched_tokens(request, 0)

        self.assertEqual(matched_tokens, 16)
        self.assertFalse(is_async)

    def test_get_num_new_matched_tokens_requires_all_layer_scoped_blocks(self):
        layer_names = ["model.decoder.layers.0.self_attn.attn", "model.decoder.layers.1.self_attn.attn"]
        connector = self.make_scheduler_connector_with_layers(layer_names)
        request = create_request(num_tokens=32, block_size=16)

        block_key = connector.block_keys_from_hashes(request.block_hashes)[0]
        connector.store_client.write_block(
            connector._layer_scoped_block_key(block_key, layer_names[0]),
            b"layer0-payload",
        )

        matched_tokens, is_async = connector.get_num_new_matched_tokens(request, 0)

        self.assertEqual(matched_tokens, 0)
        self.assertFalse(is_async)

    def test_get_num_new_matched_tokens_leaves_last_token_for_recompute(self):
        connector = self.make_connector(KVConnectorRole.SCHEDULER)
        request = create_request(num_tokens=16, block_size=16)

        block_key = connector.block_keys_from_hashes(request.block_hashes)[0]
        connector.store_client.write_block(block_key, b"block-0")

        matched_tokens, is_async = connector.get_num_new_matched_tokens(request, 0)

        self.assertEqual(matched_tokens, 0)
        self.assertFalse(is_async)

    def test_worker_save_and_load_payloads_via_metadata(self):
        worker = self.make_connector(KVConnectorRole.WORKER)
        req_id = "req-1"
        request_metadata = CurvineRequestMetadata(
            request_id=req_id,
            operation="save",
            block_ids=[1, 2],
            block_keys=["aa", "bb"],
            token_count=32,
        )
        metadata = CurvineConnectorMetadata()
        metadata.add_request(request_metadata)
        worker.bind_connector_metadata(metadata)

        worker.save_kv_layer(
            "layer0",
            torch.empty(0),
            attn_metadata=None,
            block_payloads_by_key={"aa": b"A", "bb": b"B"},
        )
        stored_blob = worker.store_client.read_block("aa")
        header, payload = deserialize_kvblk(stored_blob)
        self.assertEqual(payload, b"A")
        self.assertEqual(header.block_size_tokens, 16)

        load_metadata = CurvineConnectorMetadata()
        load_metadata.add_request(
            CurvineRequestMetadata(
                request_id=req_id,
                operation="load",
                block_ids=[1, 2],
                block_keys=["aa", "bb"],
                token_count=32,
            )
        )
        worker.bind_connector_metadata(load_metadata)
        worker.start_load_kv(None)

        self.assertEqual(
            worker.get_loaded_block_payloads(req_id),
            {"aa": b"A", "bb": b"B"},
        )

    def test_worker_save_tensor_payload_serializes_to_kvblk(self):
        worker = self.make_connector(KVConnectorRole.WORKER)
        metadata = CurvineConnectorMetadata()
        metadata.add_request(
            CurvineRequestMetadata(
                request_id="req-tensor",
                operation="save",
                block_ids=[7],
                block_keys=["tensor-block"],
                token_count=16,
            )
        )
        worker.bind_connector_metadata(metadata)

        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        worker.save_kv_layer(
            "layer0",
            torch.empty(0),
            attn_metadata=None,
            block_tensors_by_key={"tensor-block": tensor},
        )

        header, payload = deserialize_kvblk(worker.store_client.read_block("tensor-block"))
        self.assertEqual(payload, tensor.numpy().tobytes())
        self.assertEqual(header.dtype_code, DTypeCode.FP32)
        self.assertEqual(header.page_size_bytes, len(payload))

    def test_worker_load_marks_corrupted_kvblk_as_failed(self):
        worker = self.make_connector(KVConnectorRole.WORKER)
        worker.store_client.write_block("bad", b"not-a-kvblk")

        metadata = CurvineConnectorMetadata()
        metadata.add_request(
            CurvineRequestMetadata(
                request_id="req-bad",
                operation="load",
                block_ids=[99],
                block_keys=["bad"],
                token_count=16,
            )
        )
        worker.bind_connector_metadata(metadata)

        worker.start_load_kv(None)

        self.assertEqual(worker.get_loaded_block_payloads("req-bad"), {})
        self.assertEqual(worker.get_block_ids_with_load_errors(), {99})

    def test_worker_save_kv_layer_extracts_block_from_cpu_cache(self):
        worker = self.make_connector(KVConnectorRole.WORKER)
        metadata = CurvineConnectorMetadata()
        metadata.add_request(
            CurvineRequestMetadata(
                request_id="req-layer-save",
                operation="save",
                block_ids=[1],
                block_keys=["layer-block"],
                token_count=16,
            )
        )
        worker.bind_connector_metadata(metadata)

        kv_layer = torch.arange(
            3 * 2 * 16 * 2 * 4,
            dtype=torch.float32,
        ).reshape(3, 2, 16, 2, 4)

        worker.save_kv_layer("layer0", kv_layer, attn_metadata=None)

        _, payload = deserialize_kvblk(worker.store_client.read_block("layer-block"))
        self.assertEqual(payload, kv_layer[1].contiguous().numpy().tobytes())

    def test_worker_start_load_kv_injects_block_into_registered_cache(self):
        worker = self.make_connector(KVConnectorRole.WORKER)
        kv_cache = torch.zeros((4, 2, 16, 2, 4), dtype=torch.float32)
        source_block = torch.arange(2 * 16 * 2 * 4, dtype=torch.float32).reshape(
            2, 16, 2, 4
        )
        worker.register_kv_caches({"layer0": kv_cache})

        blob = serialize_kvblk(
            KvblkHeader(
                block_key_hash=b"\x11" * 16,
                model_id_hash=1,
                tp_rank=0,
                kv_group_id=0,
                dtype_code=DTypeCode.FP32,
                block_size_tokens=16,
                page_size_bytes=source_block.numel() * source_block.element_size(),
                layer_count=1,
                num_kv_heads=2,
                head_size=4,
            ),
            source_block.contiguous().numpy().tobytes(),
        )
        worker.store_client.write_block("inject-block", blob)

        metadata = CurvineConnectorMetadata()
        metadata.add_request(
            CurvineRequestMetadata(
                request_id="req-layer-load",
                operation="load",
                block_ids=[2],
                block_keys=["inject-block"],
                token_count=16,
            )
        )
        worker.bind_connector_metadata(metadata)

        worker.start_load_kv(None)
        worker.wait_for_layer_load("layer0")

        self.assertTrue(torch.equal(kv_cache[2], source_block))
        self.assertTrue(torch.count_nonzero(kv_cache[0]).item() == 0)

    def test_worker_save_kv_layer_uses_layer_scoped_block_files(self):
        worker = self.make_connector(KVConnectorRole.WORKER)
        worker.register_kv_caches(
            {
                "layer0": torch.zeros((1, 2, 16, 2, 4), dtype=torch.float32),
                "layer1": torch.zeros((1, 2, 16, 2, 4), dtype=torch.float32),
            }
        )
        metadata = CurvineConnectorMetadata()
        metadata.add_request(
            CurvineRequestMetadata(
                request_id="req-layer-scope-save",
                operation="save",
                block_ids=[1],
                block_keys=["shared-block"],
                token_count=16,
            )
        )
        worker.bind_connector_metadata(metadata)

        layer0 = torch.arange(3 * 2 * 16 * 2 * 4, dtype=torch.float32).reshape(3, 2, 16, 2, 4)
        layer1 = (layer0 + 1000).clone()

        worker.save_kv_layer("layer0", layer0, attn_metadata=None)
        worker.save_kv_layer("layer1", layer1, attn_metadata=None)

        _, payload0 = deserialize_kvblk(worker.store_client.read_block("shared-block__layer0"))
        _, payload1 = deserialize_kvblk(worker.store_client.read_block("shared-block__layer1"))
        self.assertEqual(payload0, layer0[1].contiguous().numpy().tobytes())
        self.assertEqual(payload1, layer1[1].contiguous().numpy().tobytes())

    def test_worker_wait_for_layer_load_injects_only_requested_layer(self):
        worker = self.make_connector(KVConnectorRole.WORKER)
        layer0_cache = torch.zeros((4, 2, 16, 2, 4), dtype=torch.float32)
        layer1_cache = torch.zeros((4, 2, 16, 2, 4), dtype=torch.float32)
        worker.register_kv_caches({"layer0": layer0_cache, "layer1": layer1_cache})

        layer0_block = torch.arange(2 * 16 * 2 * 4, dtype=torch.float32).reshape(2, 16, 2, 4)
        layer1_block = (layer0_block + 500).clone()
        worker.store_client.write_block(
            "shared-load__layer0",
            serialize_kvblk(
                KvblkHeader(
                    block_key_hash=b"\x22" * 16,
                    model_id_hash=1,
                    tp_rank=0,
                    kv_group_id=0,
                    dtype_code=DTypeCode.FP32,
                    block_size_tokens=16,
                    page_size_bytes=layer0_block.numel() * layer0_block.element_size(),
                    layer_count=1,
                    num_kv_heads=2,
                    head_size=4,
                ),
                layer0_block.contiguous().numpy().tobytes(),
            ),
        )
        worker.store_client.write_block(
            "shared-load__layer1",
            serialize_kvblk(
                KvblkHeader(
                    block_key_hash=b"\x33" * 16,
                    model_id_hash=1,
                    tp_rank=0,
                    kv_group_id=0,
                    dtype_code=DTypeCode.FP32,
                    block_size_tokens=16,
                    page_size_bytes=layer1_block.numel() * layer1_block.element_size(),
                    layer_count=1,
                    num_kv_heads=2,
                    head_size=4,
                ),
                layer1_block.contiguous().numpy().tobytes(),
            ),
        )

        metadata = CurvineConnectorMetadata()
        metadata.add_request(
            CurvineRequestMetadata(
                request_id="req-layer-scope-load",
                operation="load",
                block_ids=[2],
                block_keys=["shared-load"],
                token_count=16,
            )
        )
        worker.bind_connector_metadata(metadata)

        worker.start_load_kv(None)
        self.assertEqual(torch.count_nonzero(layer0_cache).item(), 0)
        self.assertEqual(torch.count_nonzero(layer1_cache).item(), 0)

        worker.wait_for_layer_load("layer0")
        self.assertTrue(torch.equal(layer0_cache[2], layer0_block))
        self.assertEqual(torch.count_nonzero(layer1_cache).item(), 0)

        worker.wait_for_layer_load("layer1")
        self.assertTrue(torch.equal(layer1_cache[2], layer1_block))

    def test_worker_save_kv_layer_skips_incomplete_block_from_slot_mapping(self):
        worker = self.make_connector(KVConnectorRole.WORKER)
        metadata = CurvineConnectorMetadata()
        metadata.add_request(
            CurvineRequestMetadata(
                request_id="req-partial-save",
                operation="save",
                block_ids=[1],
                block_keys=["partial-block"],
                token_count=16,
            )
        )
        worker.bind_connector_metadata(metadata)

        kv_layer = torch.arange(
            3 * 2 * 2 * 16 * 4,
            dtype=torch.float32,
        ).reshape(2, 3, 2, 16, 4)
        attn_metadata = SimpleNamespace(
            slot_mapping=torch.tensor([16, 17, 18, 19, 20, 21, 22, 23], dtype=torch.int64)
        )

        worker.save_kv_layer("layer0", kv_layer, attn_metadata=attn_metadata)

        self.assertFalse(worker.store_client.exists("partial-block"))

    def test_worker_wait_for_layer_load_scatter_partial_tokens_from_slot_mapping(self):
        worker = self.make_connector(KVConnectorRole.WORKER)
        kv_cache = torch.zeros((4, 2, 16, 2, 4), dtype=torch.float32)
        worker.register_kv_caches({"layer0": kv_cache})

        source_block = torch.arange(2 * 16 * 2 * 4, dtype=torch.float32).reshape(
            2, 16, 2, 4
        )
        worker.store_client.write_block(
            "partial-load__layer0",
            serialize_kvblk(
                KvblkHeader(
                    block_key_hash=b"\x44" * 16,
                    model_id_hash=1,
                    tp_rank=0,
                    kv_group_id=0,
                    dtype_code=DTypeCode.FP32,
                    block_size_tokens=16,
                    page_size_bytes=source_block.numel() * source_block.element_size(),
                    layer_count=1,
                    num_kv_heads=2,
                    head_size=4,
                ),
                source_block.contiguous().numpy().tobytes(),
            ),
        )

        metadata = CurvineConnectorMetadata()
        metadata.add_request(
            CurvineRequestMetadata(
                request_id="req-partial-load",
                operation="load",
                block_ids=[1],
                block_keys=["partial-load"],
                token_count=16,
            )
        )
        worker.bind_connector_metadata(metadata)

        forward_context = SimpleNamespace(
            slot_mapping={"layer0": torch.tensor([18, 16, 23, 17], dtype=torch.int64)}
        )
        worker.start_load_kv(forward_context)
        worker.wait_for_layer_load("layer0")

        expected_block = torch.zeros_like(source_block)
        for local_offset in (0, 1, 2, 7):
            expected_block[:, local_offset] = source_block[:, local_offset]

        self.assertTrue(torch.equal(kv_cache[1], expected_block))
        self.assertEqual(torch.count_nonzero(kv_cache[0]).item(), 0)


if __name__ == "__main__":
    unittest.main()
