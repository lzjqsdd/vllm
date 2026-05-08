# Curvine Host Real-Model Test

This document describes how to run the Curvine cross-process KV cache test
directly on the host machine with a real model.

The test file is:

- `tests/v1/kv_connector/test_curvine_e2e.py`

The real-model case uses `facebook/opt-125m` by default and verifies:

1. The first process writes Curvine `*.kvblk` files.
2. A fresh second process reuses the saved KV cache.
3. `num_cached_tokens == 128` on the second run.

## Prerequisites

Use the workspace virtual environment and make sure vLLM is installed as a CPU
build in editable mode.

```bash
cd /root/codespace/barry/codespace/vllm

uv venv --python 3.12

export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_VARIANT=cpu \
VLLM_TARGET_DEVICE=cpu \
uv pip install --python .venv/bin/python --editable .
```

If your host environment does not already have pytest:

```bash
uv pip install --python .venv/bin/python pytest
```

You can verify that the editable package is visible:

```bash
.venv/bin/python -c "import importlib.metadata as md; print(md.version('vllm'))"
```

Before running the real-model test, verify that the CPU custom op is available:

```bash
.venv/bin/python - <<'PY'
import torch
print(hasattr(torch.ops._C, "compute_slot_mapping_kernel_impl"))
PY
```

The expected result is:

```text
True
```

If this prints `False`, the host environment does not yet have the required CPU
runtime pieces for this test.

## Run the Real-Model Test

Use a fixed Hugging Face cache path and a fixed Curvine store path so the test
is easy to inspect after completion.

```bash
cd /root/codespace/barry/codespace/vllm

export HF_HOME=/tmp/hf-cache
export VLLM_CURVINE_TEST_STORE_ROOT=/tmp/curvine-host-real-store
export VLLM_CURVINE_RUN_REAL_MODEL=1
export VLLM_CURVINE_REAL_MODEL=facebook/opt-125m
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONHASHSEED=0
export VLLM_TARGET_DEVICE=cpu
export VLLM_ENABLE_V1_MULTIPROCESSING=0

mkdir -p "$HF_HOME"
rm -rf "$VLLM_CURVINE_TEST_STORE_ROOT"

.venv/bin/python -m pytest \
  --noconftest \
  tests/v1/kv_connector/test_curvine_e2e.py \
  -q \
  -k real_model
```

## Expected Result

The expected pytest summary is:

```text
1 passed, 1 deselected
```

The test performs two separate Python process runs internally:

- Run 1: `num_cached_tokens == 0`
- Run 2: `num_cached_tokens == 128`

After the first run, Curvine should write `*.kvblk` files under:

```text
/tmp/curvine-host-real-store
```

## Inspect the Generated Files

Check the Curvine store:

```bash
find /tmp/curvine-host-real-store -name '*.kvblk'
```

Check the Hugging Face cache:

```bash
ls -la /tmp/hf-cache
```

If `HF_HOME` is not set, the default Hugging Face cache path is usually:

```text
~/.cache/huggingface
```

## Notes

- The test uses `--noconftest` on purpose, so it does not depend on the full
  repo-wide pytest fixture stack.
- `VLLM_CURVINE_TEST_STORE_ROOT` is optional. If unset, pytest uses its own
  temporary directory.
- If you want to use a different real model, only change
  `VLLM_CURVINE_REAL_MODEL`.
