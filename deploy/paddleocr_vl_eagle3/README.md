# PaddleOCR-VL-1.6 + EAGLE3 on vLLM

这是 VL-1.6 EAGLE3 服务化实验的源码导出，作为 PaddleX 的可选部署扩展维护。它包括 vLLM 目标模型适配、EAGLE3 推理优化、DocLayoutV3 整页客户端及草稿训练代码。

代码最初在独立实验目录开发，当前以独立部署目录提交。它尚未并入 PaddleX 的默认 `genai` 后端，也不表示官方 vLLM 已原生支持这个模型组合。目标模型接口的长期上游位置是 vLLM；产线接入属于 PaddleX；PaddleOCR 可以引用本目录的部署方式。

## 范围与版本

- 目标：官方 **PaddlePaddle/PaddleOCR-VL-1.6**，BF16，greedy；草稿为该版本单独训练的 **EAGLE3 K=4 dense**。
- 0.32 服务：原生视觉预处理与编码，EAGLE3 草稿 INT8 top-1，目标 BF16 验证，同步 `FULL_DECODE_ONLY` CUDA Graph。0.19 使用适合旧版草稿执行器的图配置，见下方兼容说明。
- 整页：官方 PaddleX 1.6 产线，`PP-DocLayoutV3` 检测、裁块，VLM 逐块识别，官方 Markdown/JSON 后处理。
- 已验证的历史环境：Linux x86-64、Python 3.12、单张 NVIDIA A100 80GB PCIe、vLLM **0.32.0**、PyTorch **2.13.0+cu130**、Transformers **5.19.0**。cuBLASLt 算法也按已验证版本固定，不能仅升级 vLLM 后沿用准确性结论。
- **0.19.0 使用独立兼容路径**：补齐模型校验、多模态映射与草稿上下文，使用 EAGLE3 K4、完整私有词表 INT8 草稿头和完整 proposal CUDA Graph。目标保留 BF16 权重，验证投影锁定当前 CUDA12 的原生单 token 算法，attention 保留 AR1 终点与 split 规则；另提供需显式启用的 packed FA2 构建。安装、性能对照及数值边界见 [docs/VLLM019.md](docs/VLLM019.md)。未声明 0.19.1 或其它版本受支持。

权重、数据集、教师特征、缓存和平台凭据不在 Git 中。已训练的草稿目前需要从原部署的 checkpoint 单独取得；本目录没有提供它的公开下载地址。

## 安装与资产

从本分支进入 `deploy/paddleocr_vl_eagle3` 后运行命令。服务端与 PaddleX 客户端使用两个 Python 环境：

精简 Debian/Ubuntu 容器还需为版面环境的 OpenCV 安装系统库：`apt-get update && apt-get install -y libgl1 libglib2.0-0`。

```bash
python3.12 -m venv .venv-vllm
.venv-vllm/bin/python -m pip install -r requirements-vlm.txt

python3.12 -m venv .venv-layout
.venv-layout/bin/python -m pip install -r requirements-layout.txt
```

需要与主机驱动匹配的 CUDA 13 PyTorch/vLLM 和 PaddlePaddle GPU wheel；安装包版本校验通过不等于新硬件完成数值回归。布局环境不需要安装 vLLM。这里使用独立环境中的固定 PaddleX 版本，不要求把所在仓库的开发分支安装进服务端环境。

目标模型可从官方固定 revision 下载：

```bash
.venv-vllm/bin/hf download PaddlePaddle/PaddleOCR-VL-1.6 \
  --revision c5630abae1d940eafe0697512a0325494b02ab42 \
  --local-dir models/PaddleOCR-VL-1.6
```

另行准备已训练草稿 `models/PaddleOCR-VL-1.6-EAGLE3-K4-dense/` 和官方布局模型 `models/hps_layout/PP-DocLayoutV3_infer/`。各模型目录应包含相应的配置和权重。身份校验与目录约定见 [configs/models.json](configs/models.json)，草稿结构见 [configs/draft_config.json](configs/draft_config.json)。服务启动会核对已验证目标和草稿的 SHA-256。

也可以设置 `OCR_EAGLE_TARGET`、`OCR_EAGLE_DRAFT` 和 `OCR_LAYOUT_MODEL_DIR` 指向已有资产；服务端与整页客户端必须使用一致的目标、草稿路径。

## 启动服务与整页解析

```bash
.venv-vllm/bin/python scripts/supervise_hps_vlm.py \
  --run-dir results/vlm --gpu 0 --port 18780 \
  --mode eagle3-recovery --model-version 1.6 \
  --recovery-decode-graphs int8 --hps-max-completion-tokens 8192
```

服务监听 `127.0.0.1:18780`。就绪检查为 `/health/ready`，识别接口为 `/v1/chat/completions`，模型 ID 为 `PaddleOCR-VL-1.6-0.9B`。它实现当前 PaddleX 需要的单图 OCR、公式和表格请求子集，不是通用聊天网关。一个生成请求运行，最多八个等待请求。

另一终端使用相同物理 GPU 进行整页解析：

```bash
CUDA_VISIBLE_DEVICES=0 .venv-layout/bin/python scripts/benchmark_eagle3_client.py \
  --device gpu:0 --vlm-url http://127.0.0.1:18780/v1 \
  --self-test-image /path/to/document.png --output results/document.json
```

本地解析不需要评测平台。平台注册、Token 环境变量及协议说明见 [docs/PLATFORM.md](docs/PLATFORM.md)。当前服务与客户端通过本机 HTTP 通信，跨容器网络地址需要另外适配。

## 代码位置

| 路径 | 职责 |
|---|---|
| `scripts/native_dflash_target.py` | 为 PaddleOCR-VL 补齐 `SupportsEagle3` 协议；文件名保留历史来源，导出运行路径使用 EAGLE3 |
| `scripts/hps_decode_graph_*_worker.py` | vLLM worker、decode CUDA Graph 和运行时身份检查 |
| `scripts/round3_fast_engine_locked_target.py`、`round3_attention_rows.py`、`round3_locked_lt.py` | 小批验证的目标计算与 attention 适配 |
| `scripts/round3_eagle_*head.py`、`round3_eagle_fused_int8_top1.py` | 草稿私有词表输出头与 INT8 top-1 |
| `scripts/serve_hps_vlm.py`、`serve_ocr.py`、`runtime_config.py` | HTTP 服务、请求生命周期、版本与资产配置 |
| `scripts/supervise_hps_vlm.py` | 就绪检查与同配置服务恢复 |
| `scripts/benchmark_eagle3_page.py`、`benchmark_eagle3_client.py` | DocLayoutV3 整页识别和评测协议客户端 |
| `training/` | 教师特征、草稿训练、续训及 checkpoint 导出 |

训练入口、参数和数据格式见 [docs/TRAINING.md](docs/TRAINING.md)。新训练的草稿需要独立验收，不能通过改名冒充固定的已验证 checkpoint。

## 测量与已知边界

历史记录见 [benchmarks/historical_summary.json](benchmarks/historical_summary.json)：

- **本服务配置的两页整页测试**：预热后两页耗时 4.149 s → 2.910 s，**1.426×**；144/144 个裁片请求 token 一致，8/8 对整页 Markdown 和 JSON 一致。样本很小，不能代表整份数据集。
- **另一套历史裁片研究配置**：128 裁片为 **2.424×**，384/384 请求 token 一致；24 裁片为 **1.929×**。它还用了额外视觉/预处理优化，未作为本次服务源码导出，不能把它的加速比当成本服务的整页表现。

以上是 0.32 源码整理前的历史测试。首次 0.32 导出执行 CPU、配置、接口和训练模型一致性检查，未重跑 GPU 回归或完整训练，记录见 [benchmarks/export_validation.json](benchmarks/export_validation.json)。0.19.0 首次移植记录保留在 [benchmarks/vllm019_validation.json](benchmarks/vllm019_validation.json)，后续速度优化、独立裁片及整页对照见 [benchmarks/vllm019_speed_optimization.json](benchmarks/vllm019_speed_optimization.json)。两次测试的实现和验证范围不同，不能混用结论。

当前整页客户端每个裁片先生成最多 1024 token，遇到 `length` 后对相同输入重试到 8192。仍不结束会使整页失败；重复生成可能触发这一限制。该行为保留，尚未添加异常裁片兜底。无文字页可以正常产生空 Markdown，评测平台需要区分这种结果和推理异常。

精度一致性结论限定在已记录测试，不承诺所有输入、硬件或依赖版本逐 token 一致；整页 2× 也尚未达到。

## 本地检查

```bash
.venv-vllm/bin/python -m unittest discover -s tests -p 'test_runtime*.py' -v
.venv-vllm/bin/python -m unittest discover -s tests -p 'test_hps_api.py' -v
.venv-layout/bin/python -m unittest discover -s tests -p 'test_benchmark_page*.py' -v
```

首次在新环境运行时，应先比较原版与加速端的 token、停止原因和长输出行为，再评估吞吐。权重和数据按各自许可证使用；本目录源码遵循仓库的 Apache-2.0 许可证，派生自 FlashAttention 的 heuristic 另见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
