# vLLM 0.19.0 兼容部署

本路径使用已有的 VL-1.6 目标与 EAGLE3 K4 dense 草稿，无需因推理框架降版重新训练。服务根据安装的 vLLM 版本选择 worker，0.32 路径继续保留。兼容适配限定 0.19.0、单张 A100 80GB（SM80、108 SM）、greedy、K=4 和已固定的模型组合。

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

可选的 packed FA2 内核需要额外构建，并显式传给服务。以下命令使用 CUDA 12.8.93 的 `nvcc`，下载并校验固定的官方 FA2/CUTLASS 源码，不修改安装的 vLLM：

```bash
.venv-vllm019/bin/python kernels/fa2_sm80/build.py \
  --nvcc /path/to/cuda-12.8/bin/nvcc \
  --output-dir kernels/fa2_sm80/build
CUDA_VISIBLE_DEVICES=0 .venv-vllm019/bin/python tests/gpu_vllm019_packed_attention.py \
  --library kernels/fa2_sm80/build/libpaddleocr_vl_fa2_sm80.so \
  --output results/packed-fa2-parity.json
.venv-vllm019/bin/python scripts/supervise_hps_vlm.py \
  --run-dir results/vllm019-packed --gpu 0 --port 18780 \
  --mode eagle3-recovery --hps-max-completion-tokens 8192 \
  --vllm019-packed-attention-library kernels/fa2_sm80/build/libpaddleocr_vl_fa2_sm80.so
```

该选项仅用于已固定的 B1/K4/INT8 草稿和 4096 或 12288 上下文，默认不开启。启动前核对库、构建清单和源码 hash，并将它们纳入缓存身份；Q5 计划在 target graph 捕获前分配。其余形状回原 AR1 路径。内核把 Q5 的 40 个 GQA 行放入同一原生 M64 tile，保留原 AR1 的 split 数和完整 LSE 归约树。

**Packed 路径的数值范围是有限 Q/K/V。** 已知未来位置的 V 含 NaN/Inf 时，`0×V` 可使它与逐 query 原生实现不同；运行时不扫描所有 KV 检查有限性，因此不能声称所有 BF16 位模式或所有文档均等价。组件回归和整页回归也不替代全量评测。构建、stream/存储生命周期与许可证说明见 [内核文档](../kernels/fa2_sm80/README.md)。

## 兼容处理

| 部分 | 0.19.0 路径 |
|---|---|
| 模型允许列表 | 仅对校验过的 PaddleOCR-VL + EAGLE3 使用专用配置类，其它模型仍走原验证 |
| 多模态 token | 在目标内存配置上增加 `image_token_index` 别名，值与 `image_token_id` 一致 |
| 隐藏状态 | 复用 `SupportsEagle3` 适配，捕获 `[4,10,16]` |
| 长上下文 | 草稿内存配置扩展到目标上下文，最大 12288；磁盘 checkpoint 保持原样 |
| 执行器 | 明确使用旧版 V1 runner、`runner.drafter` 和 `EagleProposer` |
| 目标计算 | BF16 权重保持不变；Q2–Q5 投影使用当前 CUDA12 原生 Q1 的 cuBLASLt 算法及归约配置，实际 target 对象标记包含输出层；FA2 按每个 query 的 AR1 causal 终点计算 |
| 草稿 | 完整私有词表 INT8 top-1；稳定 Q5/B1 decode 将辅助状态投影、四步草稿 forward 和 top-1 合入一张 CUDA Graph |
| CUDA Graph | `FULL_AND_PIECEWISE` 保留目标 FULL decode、原生草稿 PIECEWISE 回退；完整草稿图按上下文分桶，B1 仍补齐到 5，最多保留 8 张图 |
| 调度 | 默认 INT8 路径使用 0.19 原生 async scheduling，`max_num_seqs=1`；HTTP 仍只放行一个活跃请求 |
| 固定形状元数据 | 融合目标 AR1 的整数索引计算；缓存 B1/K4/Q5 的只读 GPU 索引；每轮草稿 token 和输出保留独立存储 |
| 目标采样 | 保留 FP32 argmax 的 NaN 与最小索引 tie 规则，两阶段归约用独立 CUDA Graph 重放，再进行整数 greedy 接受/拒绝；约束、logprobs、自定义 sampler、hooks、跨 stream 或嵌套 capture 回退原路径 |
| 图像预处理 | 保持原生 PIL resize，用参数查表复现 rescale/normalize 的逐位结果；不缓存图片或识别结果 |
| 视觉长度元数据 | 每次 encoder forward 从实际 GPU scalar 读取一次 max length，27 层共享新建的 CPU int32 scalar；保留原 FA2 算子，不跨图片缓存 |
| HTTP | 兼容旧版异常类型和不含 `spec_decode_metrics` 的输出 |
| 统计 | readiness 中 `request_metrics_available=false`，请求级计数返回 null；不把缺失统计解释为零接受率 |

不修改安装目录中的 vLLM 源码。版本、依赖、模型身份不符时拒绝启动，避免静默回落到其它执行方式。0.32 的历史加速比和逐 token 一致性结果不直接适用于此路径。

直接使用 0.19 原生批量 attention 时，高分辨率表格回归发现过与原版单 token 解码不同的输出。因此本适配把目标验证的每个 query 按 AR1 的 causal KV 终点计算。对已验证的 A100 80GB（SM80、108 SM），在一次 FA2 调用中合并这些行，并按固定版本的官方 heuristic 计算原 AR1 的 split 数；其它设备回退到逐行原生调用。安装时核对 FA2 native operator 签名，防止错误调用其它 ABI。扩大裁片及整页对照后，还发现原生 Q5 线性层与 Q1 会选择不同的 cuBLASLt 算法，导致少量空格及字符差异。因此当前 worker 同时锁定当前 CUDA12 环境实测的原生 Q1 投影算法；O/down 仍保留逐行矩阵尺寸，合为一个 strided-batch 调用。输出层按真实 target 对象识别，不能依赖 0.19 中不存在的 head `prefix`。73 个计划在加载权重后、图捕获前准备，总额外 workspace 为 4096 字节。Q1 与大于 5 token 的图像 prefill 保留原生投影；极短纯文本 prefill 不在此图像 HTTP profile 的范围内。数值回归仍不构成所有输入逐 token 相等的证明。

0.19 的 `FULL_DECODE_ONLY` 会让草稿 forward 也处于图外，因此这里使用官方 `FULL_AND_PIECEWISE`。就绪信息会校验并报告 `draft_cudagraph_mode=PIECEWISE`。旧版可能仍打印“私有词表使 local argmax 回落”的泛化警告；本适配已替换草稿的 `get_top_tokens` 并准备 INT8 计划，该日志本身不代表此适配回落。

完整草稿图按运行时 CPU 上的乐观上下文长度加 3 选择上界，覆盖四步草稿中的后续位置；实际 causal 长度、KV 槽位和 M-RoPE 第 0 轴仍由当前请求的 GPU 张量决定。分桶仅改变草稿执行，不改变目标 attention 的 split 数。首次遇到一个分桶时需要捕获，后续请求复用图结构，每次复制当前输入并重新计算。非 Q5、prefill、非标准配置保留原生或三步图回退。就绪时图可能还没有被请求触发，不能把启动计数当作运行中的 replay 数。

`scripts/profile_vllm019.py` 可以在同一模型配置下执行预热、无 profiler 控制组、分阶段 CUDA events 和单独的 PyTorch trace，前后读取 worker 计数，确认完整草稿图确实被使用。它输出聚合耗时，不保存识别内容；trace 会带 profiling 开销，性能结论以独立 HTTP 对照为准：

```bash
.venv-vllm019/bin/python scripts/profile_vllm019.py \
  --gpu 0 --manifest /path/to/manifest.jsonl \
  --vllm019-packed-attention-library kernels/fa2_sm80/build/libpaddleocr_vl_fa2_sm80.so \
  --output results/vllm019-profile.json
```

最终配置的独立 profile 中，4 个 OCR/表格请求生成 1583 token，草稿接受率为 63.08%，每轮平均推进 3.523 token。分阶段 events 记录：视觉编码约 159 ms，target prefill 约 40 ms，target decode forward 约 810 ms，decode 输出层约 69 ms，草稿阶段约 266 ms，目标采样约 11 ms；主要瓶颈仍在 decode。统计包含 stream 等待，父子区间不能相加。无 events 的控制组为 1560 ms，带 events 为 1689 ms，说明 profiling 本身有开销。运行后计数确认四个草稿图分桶共重放 1358 次、采样图调用 1358 次，视觉 13 次 forward 各只读一次 GPU scalar，并复用其余 26 层。

## 验证

当前 A100 80GB PCIe 的配对实测，使用上述显式 packed FA2 选项、完整 40k 草稿头、采样图和视觉 scalar 复用（相同图片、greedy、上下文 12288、关闭图片/前缀结果缓存）：

| 范围 | 原版 | 当前 K4 | 加速 | 一致性 |
|---|---:|---:|---:|---|
| 32 个裁片，重复 3 次 | 39.38 s | 20.03 s | **1.97×** | 96/96 token、停止原因、输入 token 数一致 |
| 另一组 32 个裁片，1 轮 | 9.12 s | 4.96 s | **1.84×** | 32/32 token、停止原因、输入 token 数一致 |
| DocLayoutV3 + VLM + 官方后处理，4 页 × 2 轮 | 35.01 s | 26.38 s | **1.33×** | 8/8 完整结果一致 |

第一个裁片集合按任务分组，OCR 为 1.78×、公式为 1.65×、表格为 2.11×；第二个集合的表格为 2.28×。另一个高分辨率表格用例为 3698 个输入 token + 1089 个输出 token，与原版逐 token 一致。12 组原生 attention、405 组当前权重线性层 eager/graph 对照、重建 packed 库的 3656 组逐位对照、308 组 graph argmax 和 30 组视觉 scalar 对照全部通过，元数据、greedy sampler、输入拷贝和图像预处理的独立检查也通过。原生 Q1/Q5 线性层本身在相同检查中并非完全一致；当前投影修正用于消除这个已复现的问题。

本轮保留了各阶段记录：未加 packed/采样图/视觉复用的精度对齐 K4 为 1.76×；packed K4 为 1.89×；packed K6 为 1.90×。进一步尝试训练数据选出的 16k/8k 草稿词表、后三步 Q1 草稿执行，分别只有约 1.88×/1.84×/1.86×，均没有优于最终完整词表 K4。它们只改变草稿执行且测试输出一致，但未纳入服务。未修正目标线性层时曾测到约 1.82×，另一组裁片及整页出现差异，该数字不作为精度对齐版本的性能结论。

上述裁片与整页是不同口径。记录排除了首条预热请求，保留后续首次图分桶捕获的成本；长表格只作为正确性门槛，不用冷/热启动混比测速。测试时其它作业仍占用显存，没有独占 GPU。**当前整体尚未达到 2×，也未完成整个评测集的精度验收。**

另做双方充分预热后的三轮复测，每轮仍为同一组 32 裁片 × 3 次：原版三轮为 39.384/39.190/39.210 s，加速版为 19.931/19.807/19.836 s，中位数比为 **1.977×**。加速版额外 288 次请求的 token、停止原因和输入 token 数均一致。这是重复测量，不增加独立文档数，也没有稳定超过 2×。

CPU 回归在 0.19 与 0.32 环境各通过 169 项；独立构建合同 12 项、布局环境恢复合同 9 项通过。本轮没有重新执行 0.32 的 GPU 性能测试。

```bash
.venv-vllm019/bin/python -m unittest discover -s tests -v
```

空闲 A100 上可单独运行 attention 的 GPU 数值对照（不会加载模型权重）：

```bash
CUDA_VISIBLE_DEVICES=0 .venv-vllm019/bin/python tests/gpu_vllm019_attention.py \
  --output results/fa2-parity.json
```

本轮验证与限制见 [../benchmarks/vllm019_speed_optimization.json](../benchmarks/vllm019_speed_optimization.json)；首次兼容移植的历史记录保留在 `vllm019_validation.json`。对新的数据和硬件，应在 0.19.0 环境分别启动 `--mode baseline` 与 `--mode eagle3-recovery`，使用相同图片、greedy 参数及输出上限，对照 token、停止原因与耗时。不同框架版本之间的数值差异不能仅由投机解码保证消除。

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
