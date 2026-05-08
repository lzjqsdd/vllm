#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vllm.platforms as platforms
from vllm.platforms.cpu import CpuPlatform

PROMPT_IDS = list(range(256))
EXPECTED_CACHED_TOKENS = 128
REAL_MODEL_NAME = os.environ.get("VLLM_CURVINE_REAL_MODEL", "facebook/opt-125m")

pytestmark = pytest.mark.cpu_test


def _ensure_cpu_platform_for_source_tree_execution() -> None:
    if os.environ.get("VLLM_TARGET_DEVICE") != "cpu":
        return

    try:
        importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        platforms.current_platform = CpuPlatform()


_ensure_cpu_platform_for_source_tree_execution()

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.curvine.store import (
    KVBLK_FILE_SUFFIX,
)


def _write_cpu_supported_opt_config(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "_name_or_path": "minimal-opt-cpu-supported",
        "architectures": ["OPTForCausalLM"],
        "bos_token_id": 0,
        "do_layer_norm_before": True,
        "dropout": 0.0,
        "enable_bias": True,
        "eos_token_id": 2,
        "ffn_dim": 512,
        "hidden_size": 128,
        "init_std": 0.02,
        "layerdrop": 0.0,
        "max_position_embeddings": 512,
        "model_type": "opt",
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "torch_dtype": "float32",
        "vocab_size": 50272,
        "word_embed_proj_dim": 128,
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _build_llm(model: str, store_root: str, model_id: str, use_dummy: bool) -> LLM:
    init_kwargs = dict(
        model=model,
        skip_tokenizer_init=True,
        dtype="float32",
        enforce_eager=True,
        max_model_len=320,
        distributed_executor_backend="uni",
        kv_transfer_config=KVTransferConfig(
            kv_connector="CurvineKVConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "curvine_store_root": store_root,
                "curvine_model_id": model_id,
                "curvine_tp_rank": 0,
                "curvine_kv_group_id": 0,
            },
        ),
    )
    if use_dummy:
        init_kwargs["load_format"] = "dummy"
    return LLM(**init_kwargs)


def _run_curvine_once(model: str, store_root: str, model_id: str, use_dummy: bool) -> dict:
    llm = _build_llm(model=model, store_root=store_root, model_id=model_id, use_dummy=use_dummy)
    prompt = TokensPrompt(prompt_token_ids=PROMPT_IDS)
    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=1),
        use_tqdm=False,
    )
    out = outputs[0]
    return {
        "token_ids": out.outputs[0].token_ids,
        "num_cached_tokens": out.num_cached_tokens,
    }


def _run_in_fresh_process(
    model: str,
    store_root: Path,
    model_id: str,
    use_dummy: bool,
) -> dict:
    result_path = store_root / "result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        __file__,
        "--run-once",
        "--model",
        model,
        "--store-root",
        str(store_root),
        "--model-id",
        model_id,
        "--result-path",
        str(result_path),
    ]
    if use_dummy:
        cmd.append("--use-dummy")

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["VLLM_TARGET_DEVICE"] = "cpu"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    completed = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.stdout:
        sys.stderr.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    return json.loads(result_path.read_text(encoding="utf-8"))


def _assert_kvblk_files_exist(store_root: Path) -> None:
    kvblk_files = sorted(store_root.rglob(f"*{KVBLK_FILE_SUFFIX}"))
    assert kvblk_files, f"No {KVBLK_FILE_SUFFIX} files found under {store_root}"


def _resolve_store_root(default_root: Path) -> Path:
    configured_root = os.environ.get("VLLM_CURVINE_TEST_STORE_ROOT")
    if configured_root:
        return Path(configured_root)
    return default_root


def test_curvine_dummy_model_cross_process_cache_hit(tmp_path: Path) -> None:
    model_dir = tmp_path / "minimal_opt_cpu_supported"
    store_root = _resolve_store_root(tmp_path / "curvine_dummy_store")
    shutil.rmtree(store_root, ignore_errors=True)
    _write_cpu_supported_opt_config(model_dir)

    first = _run_in_fresh_process(
        model=str(model_dir),
        store_root=store_root,
        model_id="dummy-curvine-test",
        use_dummy=True,
    )
    assert first["num_cached_tokens"] == 0
    _assert_kvblk_files_exist(store_root)

    second = _run_in_fresh_process(
        model=str(model_dir),
        store_root=store_root,
        model_id="dummy-curvine-test",
        use_dummy=True,
    )
    assert second["num_cached_tokens"] == EXPECTED_CACHED_TOKENS


@pytest.mark.slow_test
@pytest.mark.skipif(
    os.environ.get("VLLM_CURVINE_RUN_REAL_MODEL") != "1",
    reason="Set VLLM_CURVINE_RUN_REAL_MODEL=1 to run the real-model Curvine test.",
)
def test_curvine_real_model_cross_process_cache_hit(tmp_path: Path) -> None:
    store_root = _resolve_store_root(tmp_path / "curvine_real_store")
    shutil.rmtree(store_root, ignore_errors=True)

    first = _run_in_fresh_process(
        model=REAL_MODEL_NAME,
        store_root=store_root,
        model_id="real-curvine-test",
        use_dummy=False,
    )
    assert first["num_cached_tokens"] == 0
    _assert_kvblk_files_exist(store_root)

    second = _run_in_fresh_process(
        model=REAL_MODEL_NAME,
        store_root=store_root,
        model_id="real-curvine-test",
        use_dummy=False,
    )
    assert second["num_cached_tokens"] == EXPECTED_CACHED_TOKENS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Curvine e2e generation pass.")
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--model", required=True)
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--use-dummy", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.run_once:
        return 1

    result = _run_curvine_once(
        model=args.model,
        store_root=args.store_root,
        model_id=args.model_id,
        use_dummy=args.use_dummy,
    )
    Path(args.result_path).write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
