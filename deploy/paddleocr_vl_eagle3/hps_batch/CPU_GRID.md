# CPU-grid 放置实验

`--cpu-grid` / `PADDLEX_HPS_CPU_GRID=1` 默认关闭，只将原 multimodal processor 的 `image_grid_thw` 字段标为 `keep_on_cpu=True`。像素仍进入 GPU；模型方法、视觉 batching、投影 GEMM 的形状和调用顺序、dtype、权重和解码设置保持不变。该选项可用于 baseline 或 EAGLE3。

当前实验未显示显著端到端收益。A100 上的 VL-1.8 + DocLayoutV4、原镜像、原 PaddleX batching 插件、OXR 100 页对照中，CPU-grid 两轮分别为 40.996 / 37.184 秒，每轮 100 页成功、1936 个裁片；均值 39.090 秒，与原配置均值 39.133 秒接近。这些结果不支持宣称已提速，更不支持两倍加速。

## 动机与验证范围

原 baseline profile 在视觉路径中观察到 7744 次 device-to-pinned-host 拷贝（1936 裁片 × 4）和 374 次 device-to-pageable-host 拷贝。已有 batching 实现将 grid 复制到 Python 来确定视觉分段，却继续把 GPU grid 传给逐图 projector；后者读取形状标量可能触发同步。CPU-grid 避免将这些仅用于形状的整数元数据搬上 GPU。

该 profile 中 `cudaStreamSynchronize` 的 CPU 累计跨度约为 7.353 秒。这包含等待已有 GPU 计算的时间，不能加到 GPU kernel 时间上，也不能视为必然可消除的端到端耗时。profile 采集开销和可能并行的诊断任务使它不适合作为正式速度成绩。

原 vLLM 0.19.0 / Torch cu129 镜像中的 8 项 CPU 检查通过，包括真实 multimodal collation 的设备路由。随后在独立 A100 上使用真实 VL-1.8 权重、原批量视觉插件，对 32 个 holdout 裁片分别以 B1、B4、B8 比较 GPU-grid 和 CPU-grid：96/96 对 vision + projector embedding 满足 `torch.equal`，最大绝对差为 0。此结果覆盖这些输入与 batch 形状，不等于对任意输入的证明，也不替代完整 OCR 精度评测。

## 启用与兼容边界

沿用 [HPS 主说明](README.md) 的原镜像、挂载与入口。Compose 可设置：

```bash
export PADDLEX_HPS_CPU_GRID=1
export PADDLEX_HPS_BATCH_VISION=0
export EAGLE3_RUN_ID=cpu-grid-baseline-001
# 使用原 compose 命令启动新的服务进程；其余配置保持原值。
```

直接 launcher 使用 `--cpu-grid`；`--no-cpu-grid` 显式关闭。该功能通过现有 `paddlex_eagle3_hps` general-plugin entry point 加载，不新增插件允许列表要求，不清空或移除 `VLLM_PLUGINS` 中的已有名称。修改开关后必须重建服务进程，不能在 processor 已实例化后热切换。

lazy model resolution 先取得所有插件注册后的实际模型类，再给该类设置一份新的 processor factory。原生父类、兄弟类、processing-info 和 dummy-input factory 不变。当前仅接受 vLLM 0.19.0 的原生 PaddleOCR processor 与已审计的原生/`_vllm_batched` 图像处理方法，使用源码 AST hash 检查；未知模型、processor 或被修改的方法会拒绝启用。此实现没有承诺兼容任意 vLLM/PaddleX 版本。独立 `--batch-vision` bridge 会修改图像处理方法，尚未与 CPU-grid 联合验证，不能同时开启；既有 PaddleX `_vllm_batched` 插件可以继续使用。

启动后检查 `*.workers/model-*.json` 中实际 `identity.cpu_grid_enabled=true`，以及 `*.workers/cpu_grid/cpu-grid-*.json` 的最终 processor 和源码身份。后者只记录类型和开关，不保存输入或 embedding。benchmark 比较会核对实际 factory 是否匹配 launch 配置，并拒绝 CPU-grid 状态不同的 EAGLE3 对照，避免混入另一项优化的影响。

## CPU 检查

在已有 vLLM 0.19.0 环境中运行，不需要模型权重或 GPU：

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover \
  -s /opt/paddlex-eagle3/tests -p 'test_cpu_grid.py' -v
```

检查覆盖真实字段 collation、原 factory 和模型方法保持不变、未知定制 processor 拒绝、注册时机和默认关闭行为。没有对应 vLLM 版本时，这组原生测试会跳过；跳过不代表验证通过。
