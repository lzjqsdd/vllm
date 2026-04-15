# Disaggregated Prefilling（实验性）

本页介绍 vLLM 中的 disaggregated prefilling 功能。

!!! note
    该功能目前仍处于实验阶段，后续可能发生变化。

## 为什么要做 disaggregated prefilling？

主要有两个原因：

- **将首 token 延迟（TTFT）和 token 间延迟（ITL）分开调优**。disaggregated prefilling 会把 LLM 推理中的 prefill 阶段和 decode 阶段放在不同的 vLLM 实例中执行。这样你就可以更灵活地给它们分配不同的并行策略（例如 `tp` 和 `pp`），在不影响 ITL 的情况下调优 TTFT，或者在不影响 TTFT 的情况下调优 ITL。
- **控制尾部 ITL**。如果不做 disaggregated prefilling，vLLM 可能会在某个请求的 decode 过程中插入新的 prefill 任务，从而拉高尾延迟。disaggregated prefilling 可以更稳定地解决这个问题，帮助你控制尾部 ITL。虽然合理设置 chunk size 的 chunked prefill 也能达到类似效果，但在实际场景里很难确定最合适的 chunk size，因此 disaggregated prefilling 通常是更可靠的方案。

!!! note
    Disaggregated prefill **不会**提升吞吐量。

## 使用示例

可以参考 [examples/online_serving/disaggregated_prefill.sh](../../examples/online_serving/disaggregated_prefill.sh) 查看 disaggregated prefilling 的使用示例。

当前支持多种 connector：

- **ExampleConnector**：可参考 [examples/offline_inference/disaggregated-prefill-v1/run.sh](../../examples/offline_inference/disaggregated-prefill-v1/run.sh) 查看 ExampleConnector 的 disaggregated prefilling 示例。
- **LMCacheConnectorV1**：可参考 [examples/others/lmcache/disagg_prefill_lmcache_v1/disagg_example_nixl.sh](../../examples/others/lmcache/disagg_prefill_lmcache_v1/disagg_example_nixl.sh) 查看 LMCacheConnectorV1 的示例。该方案底层使用 NIXL 传输 KV。
- **NixlConnector**：可参考 [tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh](../../tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh) 查看 NixlConnector 的示例。它支持完全异步的 send/recv。更详细的使用方式可见 [NixlConnector Usage Guide](nixl_connector_usage.md)，兼容性说明可见 [NixlConnector Compatibility Matrix](nixl_connector_compatibility.md)。
- **P2pNcclConnector**：可参考 [examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_example_p2p_nccl_xpyd.sh](../../examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_example_p2p_nccl_xpyd.sh) 查看 P2pNcclConnector 的使用示例。
- **MooncakeConnector**：可参考 [examples/online_serving/disaggregated_serving/mooncake_connector/run_mooncake_connector.sh](../../examples/online_serving/disaggregated_serving/mooncake_connector/run_mooncake_connector.sh) 查看 MooncakeConnector 的使用示例。更详细的使用方式可见 [MooncakeConnector Usage Guide](mooncake_connector_usage.md)。
- **CurvineKVConnector (PoC)**：这是一个仍在开发中的实验性外部文件型 KV connector。当前的设计、范围、进度和 CPU PoC 状态可参考 [Curvine Connector CPU PoC](curvine_connector_poc.md)。
- **MultiConnector**：可以利用 `KVTransferConfig` 中已有的 `kv_connector_extra_config: dict[str, Any]`，把多个 connector 按顺序放进同一个 kwargs 列表中。例如：

  ```bash
  --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_both"},{"kv_connector":"ExampleConnector","kv_role":"kv_both","kv_connector_extra_config":{"shared_storage_path":"local_storage"}}]}}'
  ```

对于 `NixlConnector`，你还可以指定一个或多个 `NIXL_Backend`，例如：

  ```bash
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both", "kv_buffer_device":"cuda", "kv_connector_extra_config":{"backends":["UCX", "GDS"]}}'
  ```

- **OffloadingConnector**：可以把 KV 数据卸载到 CPU 内存，并且自定义 CPU block size（按 token 数）以及分配的 CPU 总内存字节数：

  ```bash
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"block_size": 64, "cpu_bytes_to_use": 1000000000}}'
  ```

- **FlexKVConnectorV1**：可参考 [examples/offline_inference/prefix_caching_flexkv.py](../../examples/offline_inference/prefix_caching_flexkv.py) 查看 FlexKVConnectorV1 的使用示例。FlexKV 是一个面向超大规模 LLM 推理的分布式 KV Store 与多级缓存管理系统。

  ```bash
  --kv-transfer-config '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}'
  ```

## 基准测试

可以参考 [benchmarks/disagg_benchmarks](../../benchmarks/disagg_benchmarks) 查看 disaggregated prefilling 相关基准测试。

## 开发说明

我们通过运行两个 vLLM 实例来实现 disaggregated prefilling。一个负责 prefill（称为 prefill instance），另一个负责 decode（称为 decode instance），然后通过 connector 将 prefill instance 生成的 KV cache 和结果传递给 decode instance。

所有 disaggregated prefilling 的实现都位于 `vllm/distributed/kv_transfer` 目录下。

disaggregated prefilling 的核心抽象如下：

- **Connector**：允许 **kv consumer** 获取一批请求在 **kv producer** 一侧生成的 KV cache。
- **LookupBuffer**：提供两个 API：`insert` KV cache 和 `drop_select` KV cache。它们的语义类似 SQL：`insert` 用于把 KV cache 插入缓冲区，`drop_select` 用于查找满足条件的 KV cache，返回后再将其从缓冲区移除。
- **Pipe**：单向 FIFO tensor 传输管道，支持 `send_tensor` 和 `recv_tensor`。

!!! note
    `insert` 是非阻塞操作，而 `drop_select` 是阻塞操作。

下图展示了上述三个抽象之间的组织关系：

![Disaggregated prefilling abstractions](../assets/features/disagg_prefill/abstraction.jpg)

disaggregated prefilling 的整体工作流如下：

![Disaggregated prefilling workflow](../assets/features/disagg_prefill/overview.jpg)

图中的 `buffer` 对应 `LookupBuffer` 的 `insert` API，`drop_select` 对应 `LookupBuffer` 的 `drop_select` API。

现在，vLLM 中的每个进程都会有一个对应的 connector。具体来说包括：

- Scheduler connector：与 scheduler 进程位于同一进程中，负责调度 KV cache 传输操作。
- Worker connectors：位于各个 worker 进程中，负责实际执行 KV cache 传输操作。

下图展示了上述两类 connector 的组织关系：

![Disaggregated prefilling high level design](../assets/features/disagg_prefill/high_level_design.png)

下图展示了 worker connector 如何与 attention 模块配合，实现按层进行 KV cache 的存储与加载：

![Disaggregated prefilling workflow](../assets/features/disagg_prefill/workflow.png)

## 第三方贡献

disaggregated prefilling 与基础设施能力高度相关，因此 vLLM 在生产级 disaggregated prefilling 场景中依赖第三方 connector。同时，vLLM 团队也会积极评审并合并新的第三方 connector PR。

我们推荐三种实现方式：

- **Fully-customized connector**：自己实现 `Connector`，并调用第三方库完成 KV cache 的发送和接收，以及更多定制逻辑（例如修改 vLLM 的 model input 来实现定制化 prefilling 等）。这种方式控制力最强，但未来也最容易受到 vLLM 版本演进的影响。
- **Database-like connector**：自己实现 `LookupBuffer`，并支持类似 SQL 的 `insert` 和 `drop_select` API。
- **Distributed P2P connector**：自己实现 `Pipe`，并支持类似 `torch.distributed` 的 `send_tensor` 和 `recv_tensor` API。
