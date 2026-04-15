# Curvine Connector CPU PoC

本文总结了当前 vLLM 外部 KV 存储场景下的 Curvine 集成工作，方便后续贡献者在一个地方快速了解设计背景、当前实现状态、验证范围以及下一步计划。

!!! note
    当前工作仍然属于 PoC 阶段。目标是在进入生产级实现之前，先验证 connector 形态、存储抽象以及 block 序列化格式是否成立。

## 背景

当前方向是将 Curvine 适配为 vLLM 的外部 L2 KV 存储。

当前约定的实现路径如下：

1. 先做 PoC，而不是直接从 `OffloadingFirst` 开始。
2. 以 `KVConnectorBase_V1` 作为主要集成入口。
3. 先使用 Curvine FUSE 加 POSIX 文件 I/O，再保持清晰的抽象边界，方便后续把底层从 FUSE/POSIX 切换为 Curvine native client。
4. 一开始就使用稳定的 block 文件格式（`kvblk`），这样底层存储后端变化时不会影响 connector 语义。

在 vLLM 现有实现里，最接近的参考对象是 `HF3FSKVConnector`，因为它已经采用了外部文件型 KV connector 的模式。

## 设计目标

这个 PoC 主要验证三件事：

- vLLM 的 KV block 能否序列化成稳定的外部对象格式。
- scheduler 和 worker 两侧的 connector 生命周期能否通过 Curvine 后端完成 load 和 save。
- 当底层存储后端从 FUSE/POSIX 切换到 Curvine native client 时，connector 流程本身是否仍然成立。

这个 PoC 也明确不打算一次性解决所有问题：

- 不做生产级 metadata service。
- 不做复杂 manifest 层。
- 不做超出最小 connector 生命周期所需的异步传输流水线。
- 暂时不做真正的 GPU gather / scatter 路径。
- 暂时不保证多 rank / 多节点正确性。

## 当前架构

当前实现刻意拆成三层：

### `CurvineKVConnector`

这一层直接面对 vLLM。

主要职责：

- 实现 `KVConnectorBase_V1`。
- 基于 scheduler 侧的 block 状态构建 load / save plan。
- 触发 worker 侧的 save 和 load 操作。
- 将加载回来的 payload 注入到已注册的 KV cache 中。

当前实现位置：

- `vllm/distributed/kv_transfer/kv_connector/v1/curvine/connector.py`

### `kvblk`

这是 PoC 使用的稳定 block 序列化格式。

主要职责：

- 将一个逻辑 KV block 编码成一个二进制对象。
- 保留足够的 header 元数据，便于校验和未来兼容。
- 在不依赖底层存储后端的前提下，将字节流解码回 block payload。

当前实现位置：

- `vllm/distributed/kv_transfer/kv_connector/v1/curvine/kvblk.py`

### `CurvineStoreClient`

这是存储抽象层。

主要职责：

- 将 `block_key` 映射成底层存储对象路径。
- 提供 exists、read、write、delete 等操作。
- 屏蔽底层究竟是 POSIX/FUSE 还是未来的 Curvine native client。

当前实现位置：

- `vllm/distributed/kv_transfer/kv_connector/v1/curvine/store.py`

当前 PoC 的后端实现是 `PosixCurvineStoreClient`。

## `kvblk` V1 格式

当前 PoC 采用“一块一个文件”的方式存储 KV block，每个文件由定长 header 加原始 payload 字节组成。

V1 格式的重要属性包括：

- 每个 block 对应一个文件。
- 使用定长 header，方便快速校验。
- 当前 payload 直接存原始字节。
- 使用 CRC 校验 header 和 body。
- 在 payload 旁边保存 layout 和 dtype 元数据。

这个格式的设计目标是：

- `PosixCurvineStoreClient` 与未来的 Curvine native backend 可以共享完全相同的编码结果。
- 一旦校验失败，明确报为 load error，而不是静默读到脏数据。

## 当前对象布局

当前路径策略如下：

```text
<root>/<model_id>/<tp_rank>/<kv_group>/<hash_prefix>/<block_key>.kvblk
```

这样既能保持对象标识在不同后端中的稳定性，也能减少目录热点问题。

## 当前 PoC 范围

当前 PoC 以 CPU 路径优先，并且按 layer 组织。

已经在范围内的能力：

- scheduler 侧 block 命中检测。
- scheduler 侧 load / save metadata 生成。
- worker 侧通过 `save_kv_layer` 保存 block。
- worker 侧通过 `start_load_kv` 和 `wait_for_layer_load` 加载 block。
- CPU tensor 的序列化与反序列化。
- 按 layer 隔离的存储 key。
- load 侧根据 `slot_mapping` 进行部分 token scatter。

暂时有意不做的能力：

- save 侧从离散 slot 中真正做 partial-token gather。
- 真正的 GPU gather / scatter。
- `wait_for_save()` 和 `get_finished()` 上完整的异步保存完成语义。
- 多 rank / 多卡验证。
- 面向大量小文件的真实 Curvine FUSE 压测。

## 当前工作边界

当前实现边界被刻意收得很窄：

- 只实现和验证 Curvine connector 这条路径本身。
- 变更尽量只聚焦 `curvine` connector 代码、它的存储格式以及定向测试。
- 除非 Curvine connector 自身无法推进，否则不要把工作扩展到无关的 vLLM 子系统、泛化的 shared-test 重构，或者整体代码清理。

对于 CPU 单元测试，当前推荐路径也同样保持收敛：

- 尽量使用本地最小 model config fixture 来支撑 connector 测试。
- 避免让 Curvine connector 的验证依赖外部 Hugging Face 网络访问。
- 默认优先走离线、connector 聚焦的测试路径；只有当 connector 逻辑本身确实要求时，才升级到更宽的 runtime 或 model-loading 范围。

## 实现状态

`vllm` 中已经实现的部分包括：

- `CurvineKVConnector` 已注册进 connector factory。
- `CurvineRequestMetadata` 和 `CurvineConnectorMetadata` 可以承载每个请求的 load / save plan。
- `save_kv_layer()` 已支持把原始 payload 和 CPU tensor 持久化为 `kvblk`。
- `start_load_kv()` 已能构建按 layer 组织的待加载队列。
- `wait_for_layer_load(layer_name)` 已能执行延迟的按层注入。
- store key 已按 layer 隔离，避免同一逻辑 block key 在不同 layer 上冲突。
- 损坏的 `kvblk` 对象会被当作 load failure 处理。

CPU 路径已经不止停留在简单的 bytes round-trip：

- 已支持从 CPU KV tensor 中完整提取 block。
- 已支持把 block 完整注入回已注册的 CPU KV cache。
- load 路径已经消费 `ForwardContext.slot_mapping[layer_name]`。
- 如果一次只请求某个 block 的部分 token，当前 PoC 只会把这些 token scatter 回去，而不会覆盖整个 block。
- 如果 save 侧的 `slot_mapping` 只覆盖了一个 block 的部分内容，当前 PoC 会跳过持久化，避免写入局部或脏 block。

## 当前测试

当前 Curvine PoC 由以下定向单元测试覆盖：

- `tests/v1/kv_connector/unit/test_curvine_kvblk.py`
- `tests/v1/kv_connector/unit/test_curvine_store.py`
- `tests/v1/kv_connector/unit/test_curvine_connector.py`

这些测试当前覆盖了：

- `kvblk` header 以及 encode / decode 行为。
- store 路径映射与 POSIX 读写行为。
- connector factory 注册。
- scheduler 侧匹配 block 计数。
- scheduler 侧 load / save metadata 生成。
- worker 侧通过 metadata 执行 save / load。
- CPU tensor block 提取与回灌。
- 损坏 block 的处理。
- 按 layer 隔离的 save / load 行为。
- 基于 `slot_mapping` 的 load 侧部分 token scatter。

## 如何测试

当前推荐的验证路径分为两层。

### 1. 离线、connector 聚焦的回归测试

建议先跑这一层。这是验证 Curvine connector 边界最快的方式，而且不依赖外部网络。

```bash
PYTHONPATH=. .venv/bin/python -m unittest tests/v1/kv_connector/unit/test_curvine_kvblk.py -v
PYTHONPATH=. .venv/bin/python -m unittest tests/v1/kv_connector/unit/test_curvine_store.py -v
PYTHONPATH=. .venv/bin/python -m unittest tests/v1/kv_connector/unit/test_curvine_connector.py -v
```

预期结果：

- 三个测试文件全部通过。
- connector 测试路径不需要访问 Hugging Face 网络。

### 2. 面向本地 Curvine 路径的真实 vLLM 手动验证

如果你要验证真实的 `vllm` 请求路径，而不是只看单元测试，就用这一层。

使用前提如下：

- `Curvine` 当前用本地 POSIX 目录或者真实 Curvine FUSE 挂载点来表示。
- model path 是本地已存在的真实模型目录。
- 你的 CPU runtime 环境已经可以成功执行 `LLM.generate()`。
- 两次运行使用相同的 connector 配置和相同的存储根目录。

首先，准备一个干净的本地后端路径：

```bash
export CURVINE_ROOT=/tmp/curvine-kv-manual
rm -rf "$CURVINE_ROOT"
mkdir -p "$CURVINE_ROOT"
```

然后运行第一个进程，把外部 KV 存储写出来：

```bash
PYTHONPATH=. .venv/bin/python - <<'PY'
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

MODEL = "/path/to/local/model"
PROMPT = "Curvine connector manual validation. " * 128

llm = LLM(
    model=MODEL,
    device="cpu",
    dtype="float32",
    enforce_eager=True,
    max_model_len=1024,
    kv_transfer_config=KVTransferConfig(
        kv_connector="CurvineKVConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "curvine_store_root": "/tmp/curvine-kv-manual",
            "curvine_model_id": "manual-curvine-test",
            "curvine_tp_rank": 0,
            "curvine_kv_group_id": 0,
        },
    ),
)

outputs = llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=1))
print(outputs[0].outputs[0].text)
PY
```

确认外部 KV 对象已经落盘：

```bash
rg --files "$CURVINE_ROOT" | rg '\.kvblk$'
```

然后在一个新的进程里，用同样的 prompt 和同样的 connector 配置再次运行：

```bash
PYTHONPATH=. .venv/bin/python - <<'PY'
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

MODEL = "/path/to/local/model"
PROMPT = "Curvine connector manual validation. " * 128

llm = LLM(
    model=MODEL,
    device="cpu",
    dtype="float32",
    enforce_eager=True,
    max_model_len=1024,
    kv_transfer_config=KVTransferConfig(
        kv_connector="CurvineKVConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "curvine_store_root": "/tmp/curvine-kv-manual",
            "curvine_model_id": "manual-curvine-test",
            "curvine_tp_rank": 0,
            "curvine_kv_group_id": 0,
        },
    ),
)

outputs = llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=1))
print(outputs[0].outputs[0].text)
PY
```

这个手动场景里建议确认以下几点：

- 第一次运行后，会在配置的根目录下生成 `*.kvblk` 文件。
- 第二次运行使用同一份本地 Curvine 路径，并且在相同 prompt 形状下成功完成。
- connector 配置始终限制在 Curvine 这条路径内部，不需要引入无关的共享 connector 基础设施改动。

当前限制：

- 这条手动 `LLM.generate()` 路径仍然受下面提到的 CPU runtime 和 custom-op 环境约束。
- 如果环境缺少必需的 CPU extension，可能 connector 逻辑本身已经正确，但完整 runtime 路径仍然会失败。

## 环境说明

为了支持轻量级 Curvine connector 开发，仓库中已经提供：

- `requirements/curvine_connector_poc.txt`

内容如下：

```text
-r common.txt
-r kv_connectors.txt
pytest
```

它适合用来跑聚焦于 connector 的单元测试。

当前更实用的测试建议如下：

- 对于 connector 聚焦的 CPU 单测，优先使用本地最小 model config fixture，而不是远程模型名。
- 这样可以把 Curvine 的验证严格限制在 connector 行为本身，避免在 PoC 循环里引入额外的 Hugging Face 外部依赖。

## 当前 CPU 端到端状态

目前有两层 CPU 验证路径：

### 1. Connector 聚焦的 CPU 测试

这一层已经跑通，也是当前 PoC 可信度的主要来源。

### 2. 完整 `LLM.generate()` 的 CPU 端到端验证

这一层在当前工作区里还没有完全闭环。

最近一次环境排查得到的结论是：

- 使用预编译的 CPU 风格 editable install，已经可以启动 CPU engine、加载模型并进入执行阶段。
- 但 runtime 路径仍然缺少一些已编译 custom ops，例如 `torch.ops._C.compute_slot_mapping_kernel_impl`。
- 如果切换成完整源码 CPU 构建，则还要求 Python 环境使用 CPU 版 PyTorch，并通过非隔离的构建路径成功编译 vLLM 的 CPU 自定义扩展。

这意味着当前剩余的 CPU 端到端缺口，主要已经是环境和编译扩展问题，而不是 Curvine connector 的 Python 逻辑缺失。

## 风险

当前高风险点包括：

- 大量小 KV block 文件带来的 FUSE 延迟。
- 序列化后的 canonical layout 与运行时 KV layout 不一致。
- `block_key` 语义错误导致误命中或漏命中。
- 一块一文件带来的 metadata 压力。
- save 侧对 partial block 的处理仍然偏保守。

当前缓解手段包括：

- load 侧做 `kvblk` 校验。
- 保持清晰的 `CurvineStoreClient` 抽象边界。
- 先在 CPU 路径上把运行时语义收敛清楚，再进入 GPU 集成。
- 用定向测试覆盖按 layer 与按 slot 的关键行为。

## 下一步

当前建议的执行顺序如下：

1. 完成 CPU 路径上的 save 侧 partial-token gather。
2. 通过修复 CPU build 和 custom op 环境，打通完整的 CPU `LLM.generate()` 端到端路径。
3. 用真实请求跑通真正的 Curvine-backed CPU PoC。
4. 将 worker 路径从 CPU-only 语义扩展到真正的 GPU gather / scatter。
5. 验证多 rank 与压测场景。

## 快速贡献者地图

如果你要继续这项工作，建议从以下文件开始：

- `vllm/distributed/kv_transfer/kv_connector/v1/curvine/connector.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/curvine/kvblk.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/curvine/store.py`
- `tests/v1/kv_connector/unit/test_curvine_connector.py`
- `tests/v1/kv_connector/unit/test_curvine_kvblk.py`
- `tests/v1/kv_connector/unit/test_curvine_store.py`
- `requirements/curvine_connector_poc.txt`

如果需要追溯最初推动这个 PoC 的更宽设计背景，可以先看 Curvine 侧设计说明，再把面向贡献者的最新结论同步回本文档。
