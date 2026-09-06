# Work2 配置文件说明

本目录分别保存 DeepCorr300、m-DeepCorr 和 DeepCoFFEA 三个 Work2 目标的独立配置。三个启动脚本各自绑定一份配置，避免误用目标权重或覆盖其他实验目录。

## 配置一览

| 文件 | 用途 | 当前定位 |
| --- | --- | --- |
| `deepcorr_config.jsonc` | 使用冻结 DeepCorr300 反馈训练 Work2 生成器，并按 Work1 的数据划分和评估协议执行最终测试 | 已完成的 DeepCorr 主线 |
| `mdeepcorr_config.jsonc` | 使用冻结 DeepCorr100 筛选和冻结 DeepCorr700 复判组成的两阶段目标反馈 | m-DeepCorr 主线 |
| `deepcoffea_config.jsonc` | 使用冻结 Anchor/PandN 双塔和真实 Tor/Exit 配对反馈 | DeepCoFFEA 主线 |

## 当前主线配置

三份 `*_config.jsonc` 文件分别对应三个目标模型。它们支持 `//` 中文注释，并已用分隔线划分以下区域：

- 日常运行区：训练/评估模式、GPU 编号、实验名称和 checkpoint 恢复方式。
- 主要调参区：epoch、batch size、学习率、损失权重和 Work2 模块容量。
- 目标协议区：数据划分、窗口规格、目标阈值和评估参数。正式可比实验不要修改该区。
- 资源与实现区：数据集、Qwen、对应攻击模型权重和输出目录。

三个配置与启动脚本一一对应：

```bash
cd /home/xilin/work2
./run_work2_deepcorr.sh
./run_work2_mdeepcorr.sh
./run_work2_deepcoffea.sh
```

新训练使用：

```jsonc
"run_mode": "train",
"resume_from_checkpoint": ""
```

恢复训练将 `resume_from_checkpoint` 改为 `latest`。正式评估将 `run_mode` 改为 `evaluate`，并将 `resume_from_checkpoint` 改为 `best`。

每个 epoch 的完整 checkpoint 使用 `generator_epNNN_origrecX_advrecX_lossX_timeX_sizeX.pt` 命名；`latest.pt` 和 `best.pt` 仍作为稳定的恢复入口。其中 `origrec` 与 `advrec` 是训练期正样本验证 Recall，不是 Precision 或整体 Accuracy。

## mDeepCorr 配置说明

mDeepCorr 配置在整体结构上与 `deepcorr_config.jsonc` 一致，但目标协议区额外包含：

- `mdeepcorr100_model_path`：第一阶段冻结 DC100 权重；
- `mdeepcorr700_model_path`：第二阶段冻结 DC700 权重；
- `mdeepcorr_stage1_threshold`：DC100 第一阶段放行阈值，当前按 Work1 设置为 `0.01`；
- `flow_size=700`：DC700 使用的完整流量长度，产生 12 个生成窗口。

mDeepCorr 不得退化为只加载 DC700。其数据源沿用 Work1 的 `*_tordata300.pickle` 原始变长序列，并在加载后按 700 截断或补零。

## DeepCoFFEA 配置说明

DeepCoFFEA 配置声明双塔权重、Tor/Exit 长度、余弦 margin 与判定阈值。目前代码仍在完善，正式训练前需要完成独立验证协议检查。

## 历史配置整理

早期 smoke、probe、旧调参、消融和独立低-FPR配置已经删除。这些用途都可以通过复制主配置并修改 `model_id`、样本上限、step 上限或分支开关完成，不再为每个临时实验维护一份重复文件。
