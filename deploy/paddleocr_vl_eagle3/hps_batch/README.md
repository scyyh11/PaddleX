# 在既有 HPS 批处理服务中接入 EAGLE3

本目录针对 `develop_zjy` 的 PaddleX/HPS 链路，复用部署中的镜像、PaddleX 插件和原生 vLLM 调度。它不建立新的 Torch/CUDA 环境，不引入 HTTP 生成锁，不把 `max_num_seqs` 或产线并发改成 1，也不清空 `VLLM_PLUGINS`。已在原镜像、单张 A100 80GB PCIe 上完成 OXR 原始 100 页对照；目前 EAGLE3 能运行，但未实现吞吐提升，更未达到两倍加速。

目标测试是 **PaddleOCR-VL-1.8 + PP-DocLayoutV4**；对应原产线配置位于 `deploy/hps/sdk/pipelines/PaddleOCR-VL-1.8/server/pipeline_config.yaml`。实际部署已修改过的配置优先于仓库默认值。现有 VL-1.6 草稿可以作为兼容性试验的初始化，但其在 VL-1.8 上的接受率、吞吐与输出一致性必须重新测量，不能沿用旧 B1 实验的结论。

最终验收限定为 **单张 A100 上的完整 HPS 链路**：同一 100 页、版面模型、裁片请求、输入/输出预算及后处理，目标为两倍提速且 GT 精度不下降。增加 GPU 数量不计入这一目标。

样本来源是 [OXR 的冻结 100 页清单](https://github.com/SoMarkAI/OXR/blob/b2ee4ac12ed220bf7fefa74951babf0d3b05e766/docs/frozen-100-pages.json)。实际输入文件、顺序、原 HPS 配置、已有 PaddleX overlay 和原测速脚本均已核对；不能用 OXR 文档中的 VL-1.6/其他硬件成绩替代本模型的对照。

## 当前 100 页实测

原 HPS 使用 8 个 layout-parsing 实例、100 页请求并发、每页保留版面与后处理；每轮 100/100 成功、1936 个 VLM 裁片请求。以下为同一 GPU 上顺序执行、其他任务 GPU 空闲的两轮端到端秒数，均不启用 profiler。原历史记录为 38.775 / 37.828 秒；本次比较使用重新运行的基线。

| 配置 | 第 1 轮 | 第 2 轮 |
| --- | ---: | ---: |
| 原 HPS 基线 | 39.853 | 38.412 |
| 原 HPS + VL-1.6 草稿 K4 | 51.655 | 48.713 |
| 原 HPS + VL-1.6 草稿 K1 | 48.881 | 43.165 |
| 原 HPS + VL-1.6 草稿 K2 | 48.749 | 44.883 |
| 原 HPS + VL-1.8 试训草稿 K4 | 50.858 | 47.621 |
| 基线 + CPU-grid 实验 | 40.996 | 37.184 |
| 基线 + 实际 4 个 API worker | 37.775 | 34.648 |
| VL-1.8 试训草稿 K4 + 实际 4 个 API worker | 47.195 | 46.951 |
| VL-1.8 试训草稿 K4 + 实际 4 个 API worker + 原生同步调度 | 50.647 | 46.305 |
| 同步 K4 + API4 + 私有自适应 K4/K0 实验 | 45.654 | 40.938 |
| 异步 K4 + API4 + 私有重复 token 提议实验 | 44.239 | 39.062 |
| 原生 ngram_gpu K4 + API4（独立对照，无 EAGLE） | 50.426 | 50.938 |
| 新配对基线 + API4（hybrid 代码挂载、功能关闭） | 38.760 | 36.340 |
| 异步 EAGLE3 K4 + API4 + 私有完整 ngram 提议替换 | 39.459 | 36.898 |
| target-only + API4 + 每步 token 预算 4096 | 35.390 | 36.842 |
| 基线 + 实际 4 个 API worker + 私有视觉 rotary 拷贝消除实验 | 36.048 | 35.718 |
| 新配对基线 + API4（CPU-grid 代码挂载、功能关闭） | 35.819 | 36.105 |
| 同一代码 + API4 + CPU-grid | 36.290 | 34.750 |

使用原 OmniDocBench1.6 scorer、同一 100 页 GT 和官方按页汇总公式，基线两轮 Overall 为 96.2716 / 96.2730，旧草稿 K4 首轮为 96.2731，VL-1.8 试训草稿首轮为 96.3312。在这组评测中未观察到精度下降；基线自身也有 batch 浮点变化，以上不代表逐 token 相同或所有数据上的精度保证。试训使用独立来源的 3600 训练样本与 449 验证样本，没有使用这 100 页训练。

独立 profile 中，基线 GPU kernel 时间约 50.7% 为视觉编码、33.1% 为纯 decode 及输出 head、14.0% 为混合 prefill/decode，其余约 2.2%。旧草稿 K4 增加约 4.2 秒 draft GPU 工作，且未减少 target forward GPU 总量；低接受率的长尾请求是主要问题之一。该 profile 仅用于定位，不能把各请求耗时求和当作端到端耗时。API worker 修正的收益单独报告，不能归为 EAGLE3。

视觉 rotary 实验复用原 Triton 内核和 FP32 中间计算，保持原 batch/GEMM/attention；完整视觉 embedding 已通过有限样本的逐元素对照，但相对 API4 基线的端到端均值仅改善约 0.9%，尚不足以认定显著加速，因此未集成到本目录的部署选项。

私有自适应实验在低接受率长尾阶段停用草稿，待这些请求结束后恢复 K4；同配置同步 K4 对照的平均耗时由 48.476 秒降为 43.296 秒，但仍慢于 target-only API4 的 36.212 秒。审计记录两次 B1 零接受率切换、4149 次跳过 proposer 和 4147 次 q1 FULL graph 步骤，收益集中于最后单条请求。该候选首轮 GT Overall 为 96.2770，未集成到本目录。

另一项私有实验在连续 8 个已验证输出 token 相同时替换后续提议，继续运行完整草稿并由目标模型严格验证，保持原输出长度上限。平均耗时从异步 K4 对照的 47.073 秒降至 41.651 秒，首轮 GT Overall 为 96.2731；仍慢于 target-only，不得将该局部改善报告为原 HPS 吞吐提升。该候选未作为部署选项集成。

独立原生 `ngram_gpu` 对照使用默认查找窗口 5、K4 和异步调度，未加载 EAGLE 草稿。32 个独立裁片及带图像的 4096-token 强制重复功能测试均与 target-only 的 token/停止原因一致；正式 100 页首轮 GT Overall 为 96.3280，但平均耗时为 50.682 秒。独立裁片诊断中普通请求仅 3 次 FULL、571 次 PIECEWISE；功能正确并未转化为吞吐收益。该独立候选未加入本目录部署选项。

后续私有 hybrid 实验保持原 EAGLE3 异步 K4 验证路径，只在已验证的生成历史中找到完整 4-token 后续片段时替换整行提议；继续运行完整草稿，由目标模型严格验证。原镜像、UID65534、单 GPU0 的整条 HPS 已通过预热及两轮 100 页，均为 1936 裁片、无失败。候选均值 38.179 秒，新配对 target-only 基线为 37.550 秒，即 0.984 倍速度；相对此前 EAGLE3 的改善不能作为原 HPS 提速。两组首轮 GT Overall 分别为 96.2730 / 96.2737，公式和表格分数相同、文本有微小差异，未证明逐 token 不变。该候选未集成为公共部署选项。

首 token 路径的配置对照仅把 `max-num-batched-tokens` 从 131072 改为 4096，原镜像 CPU 参数流核实其余 230 项参数不变。两轮均值为 36.116 秒，相对同一新基线约 1.04 倍，首轮 GT Overall 为 96.2731，尚未证明稳定收益。包括预热在内的 3888 个裁片请求，平均调度后到首 token 的区间从 0.377 降至 0.217 秒，但排队从 0.809 增至 1.137 秒，TTFT 从 1.506 增至 1.671 秒；这些是并发请求阶段耗时，不是 GPU kernel 时间。vLLM 启动估算随预算变化，将 KV cache 从 16.61 增至 37.06 GiB，整卡采样显存峰值也从 55.21 增至 72.14 GiB。该配置不作为已成立的性能推荐。

后续单卡 API4 的完整 100 页诊断捕获了 300 个视觉批次、1936 个裁片。VLM worker 的 GPU kernel 合计 29.364 秒，其中视觉 tower 为 14.237 秒，混合 prefill/decode 的目标模型及输出 head 为 3.690 秒，纯 decode 及输出 head 为 10.373 秒；混合区间不能全部归为 LLM prefill。视觉 tower 中 GEMM 为 6.876 秒、attention 为 3.210 秒、拼接/拷贝为 1.897 秒。带 profiler 的整页耗时为 48.540 秒，只用于定位，不参与速度比较；该 worker trace 也不覆盖另一个进程中的版面模型，不能把 trace 中的间隙全部视为 GPU 空闲。

两个视觉组件候选已停止推进：动态形状单层编译在已完成的 9 种形状上仅有约 1.01–1.09 倍局部速度，输出出现数值差异，最大启动批次还触发编译器索引限制；保持原运算的 projector CUDA Graph 虽通过 18 个内容/形状的位级对照，但在 1936 个形状的独立加权序列中比 eager 慢约 4.18%。后者使用合成输入，不能当作完整 HPS 的速度或精度结果。两项候选均未作为部署选项集成。

CPU-grid 最早的完整对照仍使用单 API 进程，不能据此推断 API4 下的效果。后续用同一份代码、原 token 预算 131072，仅切换 grid 元数据所在设备，实际 API4 控制组均值为 35.962 秒，CPU-grid 组为 35.520 秒，约 1.012 倍。候选自身两轮相差 1.540 秒，大于均值改善的 0.442 秒，仍未证明稳定的端到端提速。包括预热的 3888 个裁片请求中，平均调度后到首 token 区间由 0.351 降至 0.268 秒、TTFT 由 2.608 降至 2.295 秒，两组均无抢占；这些请求区间的改善不能替代整页吞吐验收。两组首轮 GT Overall 为 96.2714 / 96.4462，本轮未观察到退化；输出仍有差异，不能将分差解释为 CPU-grid 提高了模型精度。

## 最小 Docker Compose 覆盖

`compose.hps-eagle3.yaml` 对应 [PaddleOCR HPS compose](https://github.com/scyyh11/PaddleOCR/blob/888b12d9a4ce3316b6714cdc0cbcc956e1604b5d/deploy/paddleocr_vl_docker/hps/compose.yaml) 的 `paddleocr-vlm-server` 服务。它继承原镜像、镜像 entrypoint、GPU、healthcheck 与原 `.env`，只为 VLM 命令加入 launcher 和三个挂载；其他服务和产线 batching 不变。

原 compose 的 command 是 `/bin/sh /entrypoint.sh`。bridge 继续执行这份原脚本，通过 `PATH` 中的轻量 shim 捕获它最终调用的 `paddleocr genai_server`，将参数原样交给原 PaddleOCR CLI，并在 PaddleX 已合并后端配置处加入 speculation。无需复制或修改原脚本的模型名称解析。`paddlex_genai_server` 的原镜像命令也可用同一个 bridge 包裹。

在原 HPS compose 目录中执行，保留现有 `.env` 内容：

```bash
export EAGLE3_ADAPTER_ROOT=/absolute/path/to/deploy/paddleocr_vl_eagle3
export EAGLE3_DRAFT_DIR=/absolute/path/to/existing-draft
export EAGLE3_RESULTS_DIR=/absolute/path/to/private-results
export EAGLE3_MODE=baseline EAGLE3_RUN_ID=baseline-001
export PADDLEX_HPS_BATCH_VISION=0
export PADDLEX_HPS_CPU_GRID=0
export PADDLEX_HPS_NATIVE_API_DISPATCH=0
mkdir -p "$EAGLE3_RESULTS_DIR"
chown 1000:1000 "$EAGLE3_RESULTS_DIR"
chmod 700 "$EAGLE3_RESULTS_DIR"

docker compose --env-file .env -f compose.yaml \
  -f "$EAGLE3_ADAPTER_ROOT/hps_batch/compose.hps-eagle3.yaml" config --quiet
docker compose --env-file .env -f compose.yaml \
  -f "$EAGLE3_ADAPTER_ROOT/hps_batch/compose.hps-eagle3.yaml" \
  up -d --no-deps --force-recreate paddleocr-vlm-server

# baseline 测量结束后，仅切换 mode / 本轮审计文件名，重复上一条 up 命令。
export EAGLE3_MODE=eagle3 EAGLE3_RUN_ID=eagle3-001 EAGLE3_K=4
```

开始请求前应等待 VLM healthcheck 就绪。新实例必须使用新的 `EAGLE3_RUN_ID`，避免容器重启覆盖一次测量的证据。若原部署已经覆盖了 VLM command，应将 override command 最后的 `/bin/sh /entrypoint.sh` 换成该实际 command 的完整 argv；`bridge -- paddlex_genai_server ...` 也保留全部原参数。不要同时启用旧 B1 adapter。

已检查的官方镜像以 `paddleocr` 用户（UID/GID 1000）运行，因此私有结果目录需要归该用户所有；以上 `chown` 需要宿主机相应权限。若部署覆盖过容器用户，应按实际 UID/GID 设置目录所有者，保持原镜像用户和私有权限。

## 使用原镜像启动 VLM

将整个 `deploy/paddleocr_vl_eagle3` 目录只读挂载到现有 VLM 容器，例如 `/opt/paddlex-eagle3`，并挂载独立的结果目录。`launch.py` 将其父目录和一个极小的 entry-point metadata 目录加到 `PYTHONPATH`，使 API/engine/worker/模型检查子进程均能发现额外插件；不需要在镜像中安装新依赖或修改 site-packages。

默认入口是镜像原有的 `paddlex_genai_server`。`--` 后应传入原来的完整参数。PaddleX 合并默认配置和用户配置后，launcher 仅向 EAGLE3 组增加 `speculative-config`；原有 batching、缓存、精度、API worker 数等均保持原值。

以下路径仅作示例，请保留实际部署的模型路径、参数和配置。镜像标识应记录实际运行的 digest，不要用未经核对的 `latest` 标签代替。

```bash
# 在现有镜像内执行；baseline 与 eagle3 使用完全相同的 -- 后参数。
python /opt/paddlex-eagle3/hps_batch/launch.py \
  --mode baseline --audit-output /results/baseline-launch.json \
  --image-reference 'your-existing-image@sha256:actual-digest' \
  --record-file /config/vllm.yaml --record-file /config/pipeline.yaml \
  -- --model_name PaddleOCR-VL-1.8-0.9B --model_dir /models/PaddleOCR-VL-1.8 \
  --host 0.0.0.0 --port 8080 --backend vllm --backend_config /config/vllm.yaml

# 停止本轮 baseline VLM 后，以相同资源和其余配置启动。
python /opt/paddlex-eagle3/hps_batch/launch.py \
  --mode eagle3 --draft-model /models/eagle3-draft --num-speculative-tokens 4 \
  --audit-output /results/eagle3-launch.json \
  --image-reference 'your-existing-image@sha256:actual-digest' \
  --record-file /config/vllm.yaml --record-file /config/pipeline.yaml \
  -- --model_name PaddleOCR-VL-1.8-0.9B --model_dir /models/PaddleOCR-VL-1.8 \
  --host 0.0.0.0 --port 8080 --backend vllm --backend_config /config/vllm.yaml
```

若实际部署原本直接执行 `vllm serve`，使用 `--entrypoint vllm`，并将 `serve MODEL ...` 原样放在 `--` 后。不要仅为此试验把 PaddleX 入口改为另一套未匹配配置。共享配置中不能预先包含 `speculative-config`，该字段由 mode 切换独占管理。

`--dry-run` 只记录计划、包版本、插件入口和源码身份，不导入 vLLM/Torch/PaddleX，也不启动 GPU。必须为每次服务实例使用新的 audit 文件名。运行时 audit 记录解析后的后端配置；相邻 `*.workers/` 目录记录实际加载模型类、视觉方法源码 hash 与 adapter 状态。

当 `VLLM_PLUGINS` 未设置时，保留 vLLM 的原生自动发现行为；已有非空允许列表会保留所有名称并追加 `paddlex_eagle3_hps`。显式空列表会报错，避免把已经禁用 PaddleX 的配置当作有效 HPS 集成。**发现插件入口不代表批量视觉编码已生效。** 应核对 worker audit 中实际选择的模型类与 `encode_image` / `get_multimodal_embeddings` 方法；有些 vLLM 版本已内置模型，PaddleX 注册函数可能因此跳过自己的实现。

`--batch-vision` / `PADDLEX_HPS_BATCH_VISION=1` 是独立的原生视觉 batching 兼容开关，默认关闭。需要试验时，baseline 与 EAGLE3 必须同时开启并核对实际 worker audit；比较工具会拒绝两边该开关不同的记录。不能将视觉 batching 本身的收益计为 EAGLE3 的收益，已有其他插件改过视觉实现时也不能假定此 bridge 已安装。

视觉 bridge 的 CPU 检查不代表数值一致性已经验收。旧机的一次 8 图 GPU 对照已有 1 图 embedding 不完全相同；扩大矩阵/卷积批次可能改变浮点结果。因此应先独立比较“原视觉实现 vs 视觉 bridge”的整页输出和评测精度，再把它作为两组共同基线。两组同时开启 bridge 只能隔离投机解码的差异，不能证明相对于原 37 秒配置的精度不变。

`--cpu-grid` / `PADDLEX_HPS_CPU_GRID=1` 是另一个默认关闭的实验选项，仅将 `image_grid_thw` 留在 CPU，沿用现有视觉 batching 和投影计算。32 裁片在 B1/B4/B8 下的 96 对 embedding 完全一致，但当前两轮完整 HPS 对照未显示显著加速，因此不建议作为已成立的性能优化默认启用。实现边界、实测数据及复现要求见 [CPU-grid 实验说明](CPU_GRID.md)。

## 修正 PaddleX 的 API worker 分派

已检查的镜像与 `develop_zjy` 均在 PaddleX backend 最后调用 `uvloop.run(api_server.run_server(args))`。在 vLLM 0.19 中，这个函数只启动单个 API worker；配置里的 `api-server-count: 4` 已被解析，却没有经过原生 CLI 中负责多 worker 的分派。不能据配置数字声称原 HPS 已运行 4 个 API worker。

`--native-api-dispatch` / `PADDLEX_HPS_NATIVE_API_DISPATCH=1` 是默认关闭的独立修补。launcher 克隆已加载的 PaddleX backend 函数，只将其末尾明确的调用替换为 `ServeSubcommand.cmd(args)`，沿用已有 parser、配置合并和原生 worker 管理；不改 site-packages、不重建镜像、不重写 `api-server-count`。源函数结构不匹配时拒绝套用。直接使用原生 `vllm serve` 入口本来就经过这个分派，无需该修补。

新机器应先在既有镜像中单独验证该选项、实际 API worker 数量和请求性能。baseline/EAGLE3 两组必须同开或同关，比较工具会核对该状态，收益不能归为 EAGLE3。CPU 测试覆盖真实 PaddleX 函数保持配置不变，以及原生 vLLM 的 1/4 worker 分派；上述 A100 实测也已核实实际 4 个 API 进程和两轮完整 100 页成功，其他部署仍需独立验证。

EAGLE3 兼容层为一维位置的草稿恢复绝对 token 序号，以匹配其训练位置与原生 KV slot 计算；目标模型仍使用原有多维位置，原生提议与验证循环保持不变。这项位置兼容修复也必须通过多请求、含图像和输出一致性验证。

## 保留原 HPS 的整页测量

HPS 网关、Triton/产线与后处理继续使用原配置。`benchmark.py` 只向已有 `/layout-parsing` 接口发送请求，不创建或修改 pipeline。它支持一份多页 PDF，也支持多张单页图像；manifest 的分组必须与原测速入口一致，不能把一份 100 页 PDF 擅自改成 100 次独立请求，反之亦然。

JSONL manifest 每行包含 `id`、本地 `file`、`file_type`（0=PDF、1=图像）和 `page_count`。例如一份真实的 100 页 PDF：

```json
{"id":"benchmark-document","file":"actual-100-page-document.pdf","file_type":0,"page_count":100}
```

`request-params.json` 必须来自原测速请求，包含相同的像素预算、输出上限、采样、版面与后处理选项，不包含 `file`、`fileType`、`logId`。不要因更短输出或更低分辨率获得表面的加速。`--concurrency` 是原 HTTP 请求并发，不是 VLM 内部 batch 大小；必须使用原测速值。

```bash
python /opt/paddlex-eagle3/hps_batch/benchmark.py record \
  --mode baseline --url http://existing-hps:8080/layout-parsing \
  --manifest /data/oxr-100.jsonl --expected-pages 100 \
  --request-params /config/request-params.json --pipeline-config /config/pipeline.yaml \
  --deployment-audit /results/baseline-launch.json \
  --concurrency 16 --warmup-runs 1 --repeats 3 --max-retries 0 \
  --output /results/baseline-private.json

# EAGLE3 服务就绪后使用相同参数，仅修改 mode、audit 和输出路径。
python /opt/paddlex-eagle3/hps_batch/benchmark.py record \
  --mode eagle3 --url http://existing-hps:8080/layout-parsing \
  --manifest /data/oxr-100.jsonl --expected-pages 100 \
  --request-params /config/request-params.json --pipeline-config /config/pipeline.yaml \
  --deployment-audit /results/eagle3-launch.json \
  --concurrency 16 --warmup-runs 1 --repeats 3 --max-retries 0 \
  --output /results/eagle3-private.json

python /opt/paddlex-eagle3/hps_batch/benchmark.py compare \
  --baseline /results/baseline-private.json --eagle /results/eagle3-private.json \
  --output /results/comparison-aggregate.json
```

示例中的 16 并发和预热一轮不是对 37 秒基线配置的推断。必须先用原始 OXR 样本、请求分组和参数复现 baseline，再开始对照。输入文件读取、hash 与 Base64 编码在计时前完成；计时包括线程池、HTTP、完整服务端处理、响应 JSON 解析及显式重试。若原测速计时范围不同，不能直接把这里的耗时除以 37。

私有记录包含每次响应、错误、重试和页级结果，权限为 0600；不得上传识别内容或输入身份。聚合比较不输出 OCR 文本和文件标识，报告总耗时、页/秒、P50/P95、失败和 429/重试，以及逐页 `prunedResult` 和 Markdown 文本的完全一致性。传输 logId、可视化图片 URL 不用于准确性比较，原响应仍保留；该接口不提供 token IDs，因此不会声称逐 token 一致。更快但缺失/失败的请求不计为加速，输入、参数、部署身份或重复次数不同会拒绝比较。录制还要求实际模型类 audit 已生成，核对 EAGLE3 组确实安装 adapter；两组实际 forward、logits 与视觉实现不同也会拒绝配对，避免只有环境开关却未生效的假对照。

裁片请求总数、接受率、GPU 峰值显存和容器镜像 digest 仍需要通过部署侧 metrics/监控核对；此 HTTP 工具不会根据页面块数冒充实际 VLM 请求数，也不自动证明镜像身份。两倍目标只有在复现的同一批处理基线上，完整工作负载耗时至少减半且输出/失败率验收通过后才成立。

## CPU 检查

```bash
python -m unittest discover -s /opt/paddlex-eagle3/tests -v
```

这些检查覆盖插件/环境隔离、配置差异、并发客户端及比较门槛，不替代真实镜像上的 GPU 和 OXR 整批验收。

`test_hps_batch_vllm_startup.py` 在已有 vLLM 0.19.0 环境中额外运行真实 CLI → `AsyncEngineArgs` 参数流，确认 general plugin 在构建 speculative config 前加载，并禁止 CUDA 初始化；没有该版本时自动跳过。compose 的 CPU 检查执行原 shell → shim → CLI 的完整进程链，后端使用测试替身；实际镜像健康检查仍需部署验证。
