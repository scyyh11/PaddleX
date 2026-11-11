# PaddleOCR-VL 端到端部署方案（4×L20）

- 更新日期：2026-11-10
- 机器：4 × NVIDIA L20（46 GB），驱动 535.154.05（最高支持 CUDA 12.2），2 × Xeon 8350C（128 线程）
- 模型：PP-DocLayoutV3 + PaddleOCR-VL-1.6-0.9B
- 本目录：方案的插件源码、部署脚本和 Triton 配置快照，源自开发机 `/ssd1/huangyizhan/benchmarks/`
- 状态：已在开发机部署，吞吐和显存已验证；**全量精度评测尚未做**（见第 9 节）

## 1. 结论

OmniDocBench v1.5 全部 1355 页，`maxNewTokens` 8192（未调整）：

| 配置 | GPU 数 | 并发 | 吞吐（页/秒） | 相对 baseline | P50 / P99 | GPU 利用率 | 每页能耗 |
|---|---|---|---|---|---|---|---|
| baseline：原生 PaddleX 服务（不用 HPS），vLLM 0.10.2 | 1 | 20 | 0.50 | 1.0× | 37.6 / 71.0 s | 38% | 315 J |
| HPS 默认（4 个 layout 实例），vLLM 0.10.2 | 1 | 20 | 1.38 | 2.8× | 12.7 / 71.1 s | 67% | 170 J |
| **本方案**，单卡 | 1 | 32 | **2.45** | **4.9×** | 11.5 / 38.4 s | 96% | 129 J |
| **本方案**，四卡同时 | 4 | 每卡 32 | **9.58** | **19.2×** | 11.3–11.5 / 39–68 s | 95–96% | 130 J |

- 四卡合计是单卡单独跑的 4 × 97.7%。四卡数据是四套服务同时各跑全部 1355 页、吞吐相加得到的。
- 9.58 页/秒是在 `gpu-memory-utilization 0.35` 下测的（V7）。之后加上的显存设置（KV 4 GiB + `expandable_segments`）在 V8、V9 中与对照差 ±1% 以内，单卡 2.38–2.41 页/秒，峰值显存从 38.2 GB 降到 27.9 GB。
- 各配置都只有同一页纯插画（`人教新起点英语5A电子课本_page_004.png`）没检出版面框，HTTP 失败为 0。
- 现在的瓶颈是 GPU 本身（利用率约 95%，功耗约 310 W / 350 W），其中视觉编码器约占 60% 的 GPU 时间。

## 2. 架构

每张 GPU 一套完全独立的服务，互不共享：

```
客户端 ──HTTP──> 网关（FastAPI，4 个 uvicorn worker）
                   │ gRPC
                   v
                 Triton：layout-parsing × 10 实例（同一张 GPU）
                   │  PP-DocLayoutV3 版面检测 → 裁剪 → 并发调用 VLM → 合并块 / Markdown
                   │ HTTP（OpenAI Chat 接口）
                   v
                 vLLM 0.19.1（同一张 GPU）+ l20_paddleocr_vl 插件
```

| GPU | 网关端口 | 网关容器 | Triton 端口（HTTP/gRPC/metrics） | Triton 容器 | Triton 部署目录 | VLM 地址 |
|---|---|---|---|---|---|---|
| 0 | 8288 | `paddleocr_hps_test_api` | 8280 / 8281 / 8282 | `paddleocr_hps_test_pipeline` | `locust_vl_20261105/hps_g0_dec_b2` | `192.168.5.3:8080` |
| 1 | 8188 | `paddleocr_hps_api_gpu1` | 8180 / 8181 / 8182 | `paddleocr_hps_gpu1` | `locust_vl_20261105/hps_g1_dec_b2` | `192.168.5.3:8081` |
| 2 | 8388 | `paddleocr_hps_api_gpu2` | 8380 / 8381 / 8382 | `paddleocr_hps_gpu2` | `locust_vl_20261105/hps_g2_dec_b2` | `192.168.5.3:8082` |
| 3 | 8488 | `paddleocr_hps_api_gpu3` | 8480 / 8481 / 8482 | `paddleocr_hps_gpu3` | `locust_vl_20261105/hps_g3_dec_b2` | `192.168.5.3:8083` |

- 路径都相对 `/ssd1/huangyizhan/benchmarks/`。
- vLLM 进程跑在 `paddleocr_dev` 容器里（bridge 网络，IP `192.168.5.3`），容器重建后 IP 可能变化。
- 目前**没有统一入口**，四个网关需要分别访问（见第 9 节）。

## 3. 配置

### 3.1 Triton（layout-parsing）

`model_repo/layout-parsing/config.pbtxt`（四张卡相同）：

```
max_batch_size: 2
instance_group [ { count: 10  kind: KIND_GPU  gpus: [ 0 ] } ]
dynamic_batching {
  preferred_batch_size: [ 2 ]
  max_queue_delay_microseconds: 20000
}
```

- 容器内 GPU 序号恒为 0，物理 GPU 由 `NVIDIA_VISIBLE_DEVICES` 指定。
- `restructure-pages` 保持 1 个 CPU 实例（跨页表格合并、标题重排，不在 `/layout-parsing` 路径上）。
- 产线 YAML（`runtime/server/pipeline_config.yaml`）与 PaddleX 官方 PaddleOCR-VL-1.6 配置一致，只改了 `VLRecognition.genai_config.server_url` 指向本卡的 VLM。
- 镜像：`sha256:19a2235b6ce9…`（paddlex 3.6 HPS 镜像），`--shm-size=4g`。

### 3.2 网关

| 变量 | 值 |
|---|---|
| `HPS_TRITON_URL` | `127.0.0.1:<本卡 gRPC 端口>` |
| `HPS_VLM_URL` | `http://192.168.5.3:<本卡 VLM 端口>` |
| `HPS_MAX_CONCURRENT_INFERENCE_REQUESTS` | 16（每个 worker；4 个 worker 共 64） |
| `HPS_INFERENCE_TIMEOUT` | 600 |

### 3.3 vLLM

- 版本：vLLM 0.19.1，torch 2.10.0+cu128，Python 3.11.10
- 环境：`vllm_versions/v0.19.1/venv311`（独立环境，不影响 `paddleocr_dev` 原有的 0.10.2）
- 实际启动命令（由 `vllm_versions/start_vllm_upstream.sh` 生成）：

```
vllm serve /ssd1/huangyizhan/cache/paddlex/official_models/PaddleOCR-VL-1.6 \
  --served-model-name PaddleOCR-VL-1.6-0.9B --trust-remote-code \
  --max-model-len 16384 --max-num-batched-tokens 131072 --api-server-count 4 \
  --chat-template /ssd1/huangyizhan/PaddleX/paddlex/inference/genai/chat_templates/PaddleOCR-VL-1.6-0.9B.jinja \
  --compilation-config.compile_mm_encoder=true --kv-cache-memory-bytes 4G \
  --host 0.0.0.0 --port <端口>
```

- 环境变量：

| 变量 | 值 | 作用 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | 本卡序号 | |
| `L20_PADDLEOCR_VL_BATCHED` | 1 | 启用插件 |
| `L20_PADDLEOCR_VL_MAX_PATCHES` | 32768 | 单次视觉编码的 patch 上限 |
| `L20_PADDLEOCR_VL_COMPILE_VIT` | 1 | 用 torch.compile 编译视觉编码器每一层（还需上面的 `compile_mm_encoder=true`） |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | 减少显存碎片 |

- `--max-model-len`、`--max-num-batched-tokens`、`--api-server-count`、聊天模板、模型名都和 PaddleX `genai_server` 的默认值一致。设了 `--kv-cache-memory-bytes` 后 `--gpu-memory-utilization` 不再生效。
- vLLM 默认开启异步调度和完整 CUDA graph（`FULL_AND_PIECEWISE`），这两项是 0.19.1 比 0.10.2 解码快约 18% 的主要原因。

### 3.4 插件 `l20_paddleocr_vl`

位置：本目录 `vllm_plugin/`，以可编辑方式安装进 vLLM 0.19.1 环境（`uv pip install -e deploy/l20_4xl20/vllm_plugin`），通过 vLLM 的 `vllm.general_plugins` 入口加载。**只有设置了 `L20_PADDLEOCR_VL_BATCHED=1` 才生效**，否则 vLLM 用自带实现。

它用一个子类替换 vLLM 自带的 `PaddleOCRVLForConditionalGeneration`（自带实现在循环里逐张编码图片），把 PaddleX 线上用过的两个优化移植过来：

1. **批量编码**（对应 PaddleX #5229）：同一调度步里的所有图片一次送进视觉编码器，每张图的注意力仍通过 `cu_seqlens` 分开。按 `L20_PADDLEOCR_VL_MAX_PATCHES` 分组，避免启动时的显存估算（104 张最大图）把 KV cache 挤没。
2. **用 CPU 端的尺寸信息**（对应 PaddleX 878b044）：`max_seqlen` 和 RoPE 表大小直接由图片尺寸算出，不再每层做 GPU 归约再 `.item()`。

其余开关（默认关闭）：`L20_PADDLEOCR_VL_COMPILE_VIT`（见上表）、`L20_PADDLEOCR_VL_ASYNC_H2D`（非阻塞拷贝，实测无收益）、`L20_PADDLEOCR_VL_TIMING`（每 10 秒打印视觉编码占用的 GPU 时间）。

### 3.5 请求参数

与 baseline 相同：`useLayoutDetection` 开，`useDocOrientationClassify`/`useDocUnwarping` 关，`mergeLayoutBlocks` 和 `prettifyMarkdown` 开，`temperature` 0，`maxNewTokens` 8192。

## 4. 部署与运维

脚本在本目录 `scripts/` 下，在宿主机上以 root 执行。机器相关的路径、容器名、镜像和 VLM 地址都在 `scripts/env.sh` 里，默认值就是开发机上的值，换机器时先 `export` 覆盖。下文 `S=deploy/l20_4xl20/scripts`。

### 4.1 准备 vLLM 环境

```bash
uv venv --python 3.11 <venv>
uv pip install --python <venv>/bin/python "vllm==0.19.1"          # 依赖 torch 2.10.0+cu128，可在驱动 535 上运行
uv pip install --python <venv>/bin/python --no-deps -e deploy/l20_4xl20/vllm_plugin
```

### 4.2 启动 VLM（每张卡一个）

```bash
VLLM_EXTRA_ARGS="--compilation-config.compile_mm_encoder=true --kv-cache-memory-bytes 4G" \
bash $S/start_vllm_upstream.sh <GPU> <端口> <venv> final-best \
  L20_PADDLEOCR_VL_BATCHED=1 L20_PADDLEOCR_VL_MAX_PATCHES=32768 L20_PADDLEOCR_VL_COMPILE_VIT=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

- 端口：GPU 0–3 依次为 8080–8083。日志在 `$L20_LOG_ROOT/<venv 所在目录名>/serve_gpu<GPU>_<端口>_final-best/server.log`。
- 每台约 90 秒就绪。**多台请依次启动**：同时启动时，几个进程会同时读写同一个 torch.compile 缓存，日志里会出现 `pickle data was truncated` 警告（不致命，会重新编译）。
- 确认插件生效：日志里应有 6 行 `will be overwritten by ... BatchedPaddleOCRVLForConditionalGeneration`，以及 `CompiledSiglipEncoderLayer`。
- 停止：`kill -TERM <vllm serve 进程号>`，可用 `ps -eo pid,cmd | grep "vllm serve .*--port <端口>"` 查到。

### 4.3 部署 Triton 和网关

```bash
# 从 PaddleX HPS SDK 生成的 PaddleOCR-VL-1.6 运行目录（含 runtime/server）派生出每张卡的目录
python3 $S/prepare_hps_variant.py hps_g<GPU>_dec_b2 --base <HPS 基础目录> --out-root <输出目录> \
  --count 10 --max-batch 2 --preferred 2 --delay-us 20000 --gpu <GPU> \
  --vlm-url http://$L20_VLM_HOST:<VLM 端口>/v1

# GPU 1–3：Triton + 网关
bash $S/deploy_gpu_stack.sh <GPU> <输出目录>/hps_g<GPU>_dec_b2 <Triton HTTP 端口> <网关端口> <VLM 端口>

# GPU 0：只换 Triton，沿用已有网关 paddleocr_hps_test_api（:8288）
bash $S/deploy_hps_variant.sh <输出目录>/hps_g0_dec_b2 <旧容器后缀>
```

- `triton/layout-parsing/config.pbtxt` 是最终方案的 Triton 配置，供核对。
- `prepare_hps_variant.py` 会去掉基础目录里测试用的只读埋点（`import hps_observer`）。
- 两个部署脚本都会先把正在运行的 Triton 容器停掉并改名保留，再用新目录起一个新容器，可以随时切回旧容器。

### 4.4 健康检查

```bash
for p in 8288 8188 8388 8488; do curl -s http://127.0.0.1:$p/health/ready; echo; done   # 网关
for p in 8080 8081 8082 8083; do curl -s -o /dev/null -w "%{http_code}\n" http://$L20_VLM_HOST:$p/health; done   # VLM
curl -s http://127.0.0.1:8280/v2/models/layout-parsing/config   # 确认 count=10、max_batch_size=2
```

### 4.5 测试前清空 prefix cache

```bash
python3 $S/flush_prefix_cache.py http://$L20_VLM_HOST:<端口> 500000 <每次不同的种子偏移>
```

## 5. 显存（每张卡，压测峰值）

| 组成 | 大小 |
|---|---|
| Triton：主进程 + 10 个 layout 实例（每个约 1.4 GB） | 约 14.4 GB |
| vLLM：权重 1.82 GiB + KV cache 4 GiB + CUDA graph 1.39 GiB + 运行时激活等 | 约 13.5 GB |
| **合计峰值** | **约 27.9 GB / 46 GB，余量约 18 GB** |

V8 对比（同一轮四卡）：对照组（`gpu-memory-utilization 0.35`）峰值 38.2 GB；只限 KV 为 4 GiB 是 30.9 GB；两项都加是 27.9 GB。吞吐差别都在 ±1.2% 以内。

## 6. 必须注意的约束

1. **Triton 的批大小必须限制。** 默认 `max_batch_size: 8`、等待 0 ms 时，并发一高 Triton 会凑出 8 页的大批次，layout 显存池涨到 GPU 装不下：V3 里峰值 45.6 GB，出现 59–91 个 HTTP 500。保持 `max_batch_size` ≤ 2（最多 4），或者限制并发。
2. **驱动 535 只支持 CUDA 12.x。** 内部 pip 源上 vLLM ≥ 0.20 依赖的 torch 是 CUDA 13 版，跑不起来。0.19.1 是源上最新的、能直接用的版本。vLLM 0.32.0 要用 wheels.vllm.ai 的 `+cu129` 包，而且还需要：把 `torchcodec` 换成 CPU 版，设 `VLLM_USE_FLASHINFER_SAMPLER=0`，加 `--trust-request-mm-kwargs`。
3. **`gpu-memory-utilization` 不是硬上限。** 0.35 时 vLLM 实际在负载下会涨到 22–23 GB。现在改用 `--kv-cache-memory-bytes` 固定 KV 大小。
4. **陷入循环的区域。** 每次全量测试有 6–9 个区域会重复输出直到 8192 个 token，负载下每个要 60–130 秒，会拖住所在的批次，是 P99 偏高的主要原因。`maxNewTokens` 不允许调整，所以这部分开销保留。
5. **四卡同时跑时，每卡比单独跑约慢 2–9%**（CPU 侧互相影响）。比较不同配置时只用同一轮里的数据。

## 7. 优化过程

| 步骤 | 改动 | 效果（同一轮对比） |
|---|---|---|
| 1 | 原生 PaddleX → HPS（4 个 layout 实例） | 0.50 → 1.38 页/秒 |
| 2 | layout 实例 4 → 6 | 1.28 → 1.57 页/秒（+23%，并发 10） |
| 3 | vLLM 0.10.2（PaddleX 插件）→ 0.19.1 + 我们的插件 | +22.8%，GPU 利用率 69% → 94% |
| 4 | 并发 32，10 个实例，每批最多 2 页 | 约 2.36 页/秒 |
| 5 | 编译视觉编码器 | +2.9%（只测 VLM：+3.4%） |
| 6 | KV 固定 4 GiB + `expandable_segments` | 峰值显存 −10 GB，吞吐不变 |

**试过但不采用：**

| 尝试 | 结果 |
|---|---|
| vLLM 0.19.1 自带实现（不加插件） | 比 0.10.2 慢 9%（逐张编码图片） |
| vLLM 0.32.0 + 插件 + 编译 | 与 0.19.1 持平（−0.2% / +1.8%），还要额外三项绕过措施 |
| FP8 在线量化（`--quantization fp8`） | 300 张裁剪图里 269 张陷入循环，输出不可用 |
| 每批最多 4 页 / 不限批大小 | 吞吐 +2% 左右，但 P99 和显存变差；不限时显存不足 |
| 12 个实例 | +1.1% |
| API 服务 4 → 8 个 | +0.4%，API 服务 CPU 都低于 42% |
| 非阻塞 H2D 拷贝 | −0.3%，阻塞只是转移到别处，引擎在等 GPU |
| 单次视觉编码 patch 上限 16384 / 65536 | −0.5% / −1.2% |
| layout 模型换 TensorRT / ONNX Runtime | layout 只占约 4% 的 GPU 时间；ONNX Runtime 更慢，TensorRT 没有生效 |
| 早期研究（10-27 到 10-29）：FastDeploy、SGLang、ngram 投机解码、async scheduling（0.10.2） | 都更慢或无收益 |

## 8. 测试方法

- 测试框架：`ocrbench_framework_20261106`（Locust 2.43.3）。产线入口发送原始页面，VLM 入口发送 PP-DocLayoutV3 实际裁出的 26,660 个区域。
- 数据：OmniDocBench v1.5 全部 1355 页，`once` 模式，每档每页只发一次，固定顺序。
- 每一轮四张卡同时跑，其中一张保持上一轮的最优配置作为对照；对照组在 V4–V9 之间波动约 0.3–1%。
- 每次测试前用随机文本把 vLLM 的 prefix cache 冲掉（`vllm_versions/flush_prefix_cache.py`），每档实测命中率 ≤ 0.2%。
- 输出一致性：每项 VLM 改动都先在 300 张裁剪图上比较输出文本，结果均为 97–99% 完全相同。作为参照，同一配置跑两次也只有约 98–99% 相同（动态批次导致的数值差异）。
- 结果目录：`ocrbench_framework_20261106/runs/`，命名为 `v<轮次>-...-20261110`。

## 9. 待办

1. **全量精度评测。** 目前只在 300 张裁剪图上比过输出，上线前需要保存每页完整 Markdown，用 OmniDocBench 评测脚本对比本方案和 baseline。
2. **统一入口。** 生产环境需要一个总入口把请求分到四个网关，本地有 nginx 1.14 镜像（`iregistry.baidu-int.com/kongming/nginx:1.14`）可以用。
3. **服务化。** 现在各服务都是手动启动的，机器重启后不会自动恢复。
4. **清理。** 崩溃留下的 `vllm_versions/v0.19.1/serve_gpu2_8082_batched/core.111533`（19 MB）；停止保留的旧 Triton 容器。

## 10. 本目录文件

| 路径（相对 `deploy/l20_4xl20/`） | 内容 |
|---|---|
| `vllm_plugin/` | vLLM 插件 `l20_paddleocr_vl`（`pyproject.toml`、`__init__.py`、`model.py`） |
| `scripts/env.sh` | 机器相关默认值（路径、容器名、镜像、VLM 地址），可用环境变量覆盖 |
| `scripts/start_vllm_upstream.sh` | 启动 vLLM（`vllm serve`），支持 `GPU_MEM_UTIL`、`API_SERVERS`、`VLLM_EXTRA_ARGS` |
| `scripts/prepare_hps_variant.py` | 从 HPS 基础目录生成每张卡的 Triton 运行目录 |
| `scripts/deploy_gpu_stack.sh` | 部署 GPU 1–3 的 Triton 和网关 |
| `scripts/deploy_hps_variant.sh` | 部署 GPU 0 的 Triton |
| `scripts/flush_prefix_cache.py` | 测试前清空 prefix cache |
| `triton/layout-parsing/config.pbtxt` | 最终方案的 layout-parsing Triton 配置 |

测试结果、日志、模型文件和虚拟环境不在本目录，仍在开发机 `/ssd1/huangyizhan/benchmarks/`（结果在 `ocrbench_framework_20261106/runs/`）。
