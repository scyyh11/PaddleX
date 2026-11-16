# VL-1.6 EAGLE3 草稿训练

这里提供本项目实际使用的训练算法、教师特征捕获和 checkpoint 导出代码。代码针对 **PaddlePaddle/PaddleOCR-VL-1.6**，不是早期 PaddleOCR-VL v1 草稿。训练数据、教师特征、中间 checkpoint 和最终权重没有随源码公开；使用者需要自行准备有权使用的数据及本地模型。来源网址和校验和不构成数据再分发许可。

`training/source_provenance.json` 记录导出前源代码的 SHA-256 和整理时的改动。导出移除了固定机器路径、GPU 2 限制及无关旧实验依赖；模型结构、损失、采样和选模计算保持原实现。新的训练会记录导出代码自身的校验和。整理后只完成 CPU 导入、模型计算一致性和命令行检查，没有重新跑完整训练；不保证重新训练得到相同权重 SHA。

## 模型与训练方式

- 目标权重固定为 VL-1.6，原始 revision 为 `c5630abae1d940eafe0697512a0325494b02ab42`，`model.safetensors` SHA-256 为 `85a479d506a11e724e7285d395c551be69f41dbc16b6342d3cacfb189aed71db`。
- 草稿是一层 Transformer：hidden size 1024、intermediate size 2048、8 个 query head、2 个 KV head、head dim 128。三个目标隐藏状态通过 `3072 → 1024` 投影后参与预测。
- 辅助状态配置 `[4, 10, 16]` 对应零基 decoder 层 `3/9/15` 后的 residual stream。特征位置 `t` 与已知 token `t+1` 用来预测 token `t+2`。
- 草稿 Transformer 有 13,635,584 个可训练参数。最终私有输出头为 `40000 × 1024`，复制目标头对应行并冻结；输入 embedding 与目标共享，不进入草稿 checkpoint。训练时的完整草稿权重约 54.60M 参数，INT8 输出头优化发生在推理加载时。
- 训练展开 4 步；后续步使用草稿自己预测的隐藏状态，训练时输入教师 token，验证时输入草稿预测 token。目标模型及其 embedding/head 冻结。
- 从零阶段使用 hard/soft CE 混合。续训阶段增加目标最终隐藏状态的 cosine alignment，以及带温度的 soft CE。默认 `--head-rank 0` 保持私有头权重冻结，历史发布 checkpoint 使用该配置。
- 验证是固定、按文档平衡的锚点上的 K=4 连续命中个数。双语料阶段用 20% 初始独立语料 + 80% 外部语料的验证分数选 checkpoint；它不是推理加速比，也不是 OCR 准确率。

## 文件入口

| 文件 | 用途 |
|---|---|
| `training/eagle_model.py` | 可训练 EAGLE3 及 vLLM 格式模型导出 |
| `training/common.py` | 模型固定校验、语料读取、锚点、展开与验证 |
| `training/capture.py`、`feature_worker.py` | vLLM 0.32.0 教师回复及真实中间特征捕获 |
| `training/train_clean.py` | 在独立 OCR/公式/表格语料上随机初始化训练 |
| `training/continue_train.py` | 一个初始语料及可选外部混合语料上的续训 |
| `training/merge_corpora.py` | 校验文档/图像隔离并链接已捕获的特征 |
| `training/retrain.sh` | 从零阶段加四轮续训的原始参数顺序 |

## 输入数据格式

每个语料目录至少包含 `manifest.jsonl`、`selection.json`、图片，以及捕获后生成的 `teacher.jsonl` 和 `features/`。`manifest.jsonl` 只含 `train`、`validation`；测试集可以单独放在 `dev_test.jsonl`，训练入口不会读取它。每个 split 都需要具有至少两个生成 token 的 OCR、公式、表格样本。

一行清单示例（示例路径及 hash 必须替换为实际值）：

```json
{"id":"ocr_document001_crop01","image":"/absolute/path/crop.png","image_sha256":"IMAGE_SHA256","document_id":"document001","split":"train","task":"ocr","prompt":"OCR:","source":"https://huggingface.co/datasets/wikimedia/wikipedia"}
```

`task` 为 `ocr` / `formula` / `table`；相应 prompt 为 `OCR:` / `Formula Recognition:` / `Table Recognition:`。ID 必须唯一，并作为特征文件名使用。文档不允许跨 split；相同图片 hash 不允许被重复加入续训语料。

`selection.json` 至少包含下面字段，hash 是最终 `manifest.jsonl` 的文件 SHA-256：

```json
{"sources":["https://huggingface.co/datasets/wikimedia/wikipedia"],"manifest_sha256":"MANIFEST_SHA256"}
```

初始独立语料的来源校验保留原训练设置：Wikipedia / Python 文档文本、程序生成的数学公式、FinTabNet_OTSL 表格。续训的第二个语料按其 `selection.json.sources` 校验，并拒绝 OmniDocBench 来源。公开代码不包含原数据下载与筛选产物；训练者须保留来源、许可和文档拆分记录。

教师特征由捕获脚本保存为 `features/<id>.pt`，包含：`token_ids`、`generated_ids`、`prompt_len`、`positions`、`features [N,1024]`、`aux_features [N,3,1024]`、`aux_layer_ids`，以及 ID、split、task、document ID、目标 revision/hash 等元信息。特征使用 BF16；加载采用 `weights_only=True`。

## 捕获教师回复和特征

使用已安装 vLLM **0.32.0**、PyTorch、Transformers、Pillow、safetensors 的环境，按根目录运行环境说明安装依赖。模型目录需包含该固定版本的 tokenizer、processor 和聊天模板。选一张空闲 GPU；以下两阶段依次运行，不与在线服务抢占 GPU。

```bash
export CUDA_VISIBLE_DEVICES=0
python training/capture.py generate --target /path/to/PaddleOCR-VL-1.6 --data-dir /path/to/corpus --max-generation 1024
python training/capture.py features --target /path/to/PaddleOCR-VL-1.6 --data-dir /path/to/corpus --max-generation 1024
python training/capture.py audit --target /path/to/PaddleOCR-VL-1.6 --data-dir /path/to/corpus --max-generation 1024
```

`generate` 使用 BF16、temperature 0、seed 0。`features` 用 eager、单请求教师强制前向；逐项核对实际 token ID、位置及特征覆盖。默认只处理 `manifest.jsonl`；需要独立捕获测试集时，三条命令均显式添加 `--include-heldout`，并保持测试特征与训练分离。

历史 `v16_clean` 和 `v16_external` 的输出上限为 512；`v16_research_extra` 和 `v16_dense` 为 1024。保持同一语料 generate/features/audit 的上限一致。`audit` 的最终隐藏状态 CPU 投影只要求抽样 top-1 一致率大于 95%；全序列前向与增量解码在 BF16 临界值处可能不同。训练 hard label 始终使用真实 vLLM 生成 token，不能把该阈值当作推理精度保证。

## 从零和续训

准备 `V16_DATA_ROOT` 下四个语料目录：`v16_clean`、`v16_external`、`v16_research_extra`、`v16_dense`，其中后两个分别是增量公式/科学表格和密集排版数据。每个目录先完成教师捕获。重跑脚本要求新的输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 \
V16_TARGET=/path/to/PaddleOCR-VL-1.6 \
V16_DATA_ROOT=/path/to/prepared-corpora \
V16_OUTPUT_ROOT=/absolute/path/to/new-training-run \
V16_PYTHON=python \
bash training/retrain.sh
```

已有同目标模型的独立草稿时，可直接续训。第一个 `--data` 必须是初始独立语料，第二个可选，最多两个；多个外部语料先用 `merge_corpora.py` 合并。

```bash
CUDA_VISIBLE_DEVICES=0 python training/continue_train.py \
  --target /path/to/PaddleOCR-VL-1.6 \
  --data /path/to/v16_clean --data /path/to/merged-external \
  --prior /path/to/prior-draft --run /path/to/new-run \
  --output /path/to/new-draft --steps 8000 --eval-every 800 \
  --validation-anchors 800 --lr 0.00014 --hard-weight 0.74 \
  --temperature 1.6 --align-weight 1.2 --external-weight 0.90 \
  --seed 20261122 --vocab-size 40000
```

草稿 prior 需要 `config.json`、`model.safetensors`、`training_metadata.json`；元信息必须声明 `omnidocbench_used=false` 并匹配固定的 VL-1.6 hash。续训还会检查私有头行与目标头一致。不同目标模型版本的草稿不能直接混用。

## 已记录的训练与结果

| 阶段 | 优化步数 | 最终选择 step | 有效训练 / 验证行 | 私有词表 |
|---|---:|---:|---:|---:|
| Clean 随机初始化 | 7200 | 7200 | 4984 / 120 | 32768 |
| Aligned | 4600 | 3680 | 4984 / 120 | 32768 |
| Real PDF | 6400 | 6400 | 9089 / 211 | 37459 |
| Research | 9600 | 9600 | 12445 / 323 | 40000 |
| Dense | 8000 | 8000 | 16613 / 449 | 40000 |

原数据来源包括 Wikipedia、FinTabNet_OTSL、程序生成公式、arXiv、HKEX 公告、Europe PMC、巨潮资讯和 Federal Register。各文档及数据集采用各自许可；本仓库不重新分发这些内容，也没有将评测集用于训练。发布前原始语料审计记录跨 split 文档重叠为 0、图像重复为 0。

最终草稿标识为 `PaddleOCR-VL-1.6-EAGLE3-K4-dense`，`model.safetensors` SHA-256：

```text
b7f5dc74168e594eea4c7d961c4c95e7f9c730adbd058f76d20ea48275a4157d
```

训练导出原始 `max_position_embeddings=4096`；当前长输出服务的运行时适配把草稿的位置长度扩到 12288，这不等于训练时使用了 12288-token 序列。服务若严格检查发布 hash，新训练的 checkpoint 不能直接冒充已验证的最终草稿；需先完成独立精度、速度与长序列回归，再单独管理其部署标识。

历史独立 128 裁片、单 A100、warm B1 测得 2.424×，384/384 请求的 token 和停止原因一致；另一组 24 裁片为 1.929×。整页 DocLayoutV3 + VLM 的两页小样本是 1.426×。上述结果对应不同的已冻结推理配置和评估范围，不是“任意整页保证 2×”，也不是这份重新训练代码的实测结果。
