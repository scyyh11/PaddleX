# vLLM 0.19.0 兼容部署

本路径使用已有的 VL-1.6 目标与 EAGLE3 K4 dense 草稿，无需因推理框架降版重新训练。服务根据安装的 vLLM 版本选择 worker，0.32 路径继续保留。兼容适配限定 0.19.0、单 GPU、greedy、K=4 和已固定的模型组合。

## 安装与启动

在 `deploy/paddleocr_vl_eagle3` 下创建独立环境，不覆盖 0.32 环境：

```bash
python3.12 -m venv .venv-vllm019
.venv-vllm019/bin/python -m pip install -r requirements-vlm019.txt
export OCR_EAGLE_TARGET=/absolute/path/to/PaddleOCR-VL-1.6
export OCR_EAGLE_DRAFT=/absolute/path/to/PaddleOCR-VL-1.6-EAGLE3-K4-dense
.venv-vllm019/bin/python scripts/supervise_hps_vlm.py \
  --run-dir results/vllm019 --gpu 0 --port 18780 \
  --mode eagle3-recovery --model-version 1.6 \
  --recovery-decode-graphs int8 --hps-max-completion-tokens 8192
```

权重身份校验与 0.32 一致；没有修改目标或草稿的磁盘文件。已有 DocLayoutV3 客户端和官方后处理继续连接 `/v1/chat/completions`，模型 ID 仍为 `PaddleOCR-VL-1.6-0.9B`，无需改评测平台协议。版面客户端仍使用根目录说明中的独立 PaddleX 环境。

默认保留 4 GiB KV cache 和 0.45 的 GPU 显存比例。需要控制显存时可传 `--kv-cache-memory-mib` 与 `--gpu-memory-utilization`，两个参数在服务端和 supervisor 中均可用；显存不足会启动失败。`--recovery-decode-graphs bf16` 可使用 BF16 草稿头做诊断。

## 兼容处理

| 部分 | 0.19.0 路径 |
|---|---|
| 模型允许列表 | 仅对校验过的 PaddleOCR-VL + EAGLE3 使用专用配置类，其它模型仍走原验证 |
| 多模态 token | 在目标内存配置上增加 `image_token_index` 别名，值与 `image_token_id` 一致 |
| 隐藏状态 | 复用 `SupportsEagle3` 适配，捕获 `[4,10,16]` |
| 长上下文 | 草稿内存配置扩展到目标上下文，最大 12288；磁盘 checkpoint 保持原样 |
| 执行器 | 明确使用旧版 V1 runner、`runner.drafter` 和 `EagleProposer` |
| 目标计算 | 原生 BF16 linear；FA2 按每个验证 query 的 AR1 causal 终点计算，不加载 CUDA13 cuBLASLt recipes / FA4 hook |
| 草稿 | 复用完整私有词表 INT8 top-1；0.19 的 top-1 调用位于 draft forward CUDA Graph 外 |
| CUDA Graph | 同步 `FULL_AND_PIECEWISE`：目标 FULL decode，草稿 PIECEWISE，捕获大小为 5；B1 草稿补齐到 5 后计算 |
| HTTP | 兼容旧版异常类型和不含 `spec_decode_metrics` 的输出 |
| 统计 | readiness 中 `request_metrics_available=false`，请求级计数返回 null；不把缺失统计解释为零接受率 |

不修改安装目录中的 vLLM 源码。版本、依赖、模型身份不符时拒绝启动，避免静默回落到其它执行方式。0.32 的历史加速比和逐 token 一致性结果不直接适用于此路径。

直接使用 0.19 原生批量 attention 时，高分辨率表格回归发现过与原版单 token 解码不同的输出。因此本适配把目标验证的每个 query 按 AR1 的 causal KV 终点计算。对已验证的 A100 80GB（SM80、108 SM），在一次 FA2 调用中合并这些行，并按固定版本的官方 heuristic 计算原 AR1 的 split 数；其它设备回退到逐行原生调用。安装时核对 FA2 native operator 签名，防止错误调用其它 ABI。此修正只对齐 attention，没有替换目标线性层，也不构成所有输入逐 token 相等的证明。

0.19 的 `FULL_DECODE_ONLY` 会让草稿 forward 也处于图外，因此这里使用官方 `FULL_AND_PIECEWISE`。就绪信息会校验并报告 `draft_cudagraph_mode=PIECEWISE`。旧版可能仍打印“私有词表使 local argmax 回落”的泛化警告；本适配已替换草稿的 `get_top_tokens` 并准备 INT8 计划，该日志本身不代表此适配回落。

## 验证

当前 A100 80GB PCIe 上的实测：32 个独立裁片重复 3 次，**96/96 次 token、停止原因及输入 token 数一致**；总耗时 **38.99 s → 28.94 s（1.35×）**。另一个高分辨率表格用例为 3698 个输入 token + 1089 个输出 token，与原版逐 token 一致。12 组独立 attention GPU 对照最大差值为 0，DocLayoutV3 整页冒烟测试完成了 18 个区块的识别和 Markdown/JSON 后处理。

上述速度是裁片请求的时间，不是整页吞吐；测试时其它作业仍占用显存，没有独占 GPU。整页只验证链路成功，未进行整页原版精度/速度对照。此版本尚未达到 2×，也未完成整个评测集的精度验收。

```bash
.venv-vllm019/bin/python -m unittest discover -s tests -v
```

空闲 A100 上可单独运行 attention 的 GPU 数值对照（不会加载模型权重）：

```bash
CUDA_VISIBLE_DEVICES=0 .venv-vllm019/bin/python tests/gpu_vllm019_attention.py \
  --output results/fa2-parity.json
```

GPU 验证记录见 [../benchmarks/vllm019_validation.json](../benchmarks/vllm019_validation.json)。对新的数据和硬件，应在 0.19.0 环境分别启动 `--mode baseline` 与 `--mode eagle3-recovery`，使用相同图片、greedy 参数及输出上限，对照 token、停止原因与耗时。不同框架版本之间的数值差异不能仅由投机解码保证消除。

对照时两端的**引擎上下文上限也必须一致**，因为 FA2 的 split heuristic 会使用该上限。与上面的 12288 上下文 EAGLE3 服务比较时，原版也应设置 `--hps-max-completion-tokens 8192`，即使测试请求本身只生成最多 1024 token：

```bash
.venv-vllm019/bin/python scripts/serve_hps_vlm.py --gpu 0 --port 18780 \
  --mode baseline --hps-max-completion-tokens 8192
```

`scripts/compare_vlm_outputs.py` 支持依次录制两个服务，避免同时加载两套权重。输入 JSONL 每行需有唯一 `id`、图片路径 `image` 和 `task`（`ocr` / `formula` / `table`）：

```bash
# 原版服务运行时录制，完成后停止原版并启动 EAGLE3 服务。
.venv-vllm019/bin/python scripts/compare_vlm_outputs.py record \
  --url http://127.0.0.1:18780/v1 --manifest /path/to/manifest.jsonl \
  --output results/baseline.json --max-tokens 1024
# EAGLE3 服务运行时录制。
.venv-vllm019/bin/python scripts/compare_vlm_outputs.py record \
  --url http://127.0.0.1:18780/v1 --manifest /path/to/manifest.jsonl \
  --output results/eagle.json --max-tokens 1024
.venv-vllm019/bin/python scripts/compare_vlm_outputs.py compare \
  --baseline results/baseline.json --eagle results/eagle.json \
  --output results/comparison.json
```

工具会排除第一条预热请求，核对图片 hash、任务、顺序和生成上限；比较结果包含逐 token、停止原因及分任务耗时。单轮时间只能作为快速检查，正式性能结论需充分预热并多轮重复。

训练与续训代码继续保留；教师特征捕获依赖 0.32 的 runner 接口，仍需使用原 0.32 环境。已有草稿用于 0.19 推理不需要重新捕获教师特征。
