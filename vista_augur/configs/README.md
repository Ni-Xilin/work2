# 配置文件说明

本目录只保留当前 Work2 主线和两个后续跨攻击模型验证所需的配置。日常 DeepCorr300 训练与调参只编辑第一份 JSONC；其余两份是后续扩展预设，不阻塞当前主线。

## 配置一览

| 文件 | 用途 | 当前定位 |
| --- | --- | --- |
| `second_workpoint_deepcorr300_work1_aligned.jsonc` | 使用冻结 DeepCorr300 反馈训练 Work2 生成器，并按 Work1 的数据划分和评估协议执行最终测试 | 当前唯一正式主线配置 |
| `second_workpoint_mdeepcorr_full.json` | 将冻结目标模型切换为 m-DeepCorr，用于后续兼容性训练或验证 | 扩展预设，尚未执行正式实验 |
| `second_workpoint_deepcoffea_full.json` | 将数据和冻结目标模型切换为 DeepCoFFEA，用于后续跨攻击模型验证 | 扩展预设，尚未执行正式实验 |

## 当前主线配置

`second_workpoint_deepcorr300_work1_aligned.jsonc` 是日常唯一需要修改的文件。它支持 `//` 中文注释，并已用分隔线划分以下区域：

- 日常运行区：训练/评估模式、GPU 编号、实验名称和 checkpoint 恢复方式。
- 主要调参区：epoch、batch size、学习率、损失权重和 Work2 模块容量。
- Work1 对齐协议区：数据划分、窗口规格、199 个负样本以及 `10^-3 / 10^-4` FPR。正式可比实验不要修改该区。
- 资源与实现区：数据集、Qwen、DeepCorr300 权重和输出目录。

运行方式固定为：

```bash
cd /home/xilin/work2
./run_work2.sh
```

新训练使用：

```jsonc
"run_mode": "train",
"resume_from_checkpoint": ""
```

恢复训练将 `resume_from_checkpoint` 改为 `latest`。正式评估将 `run_mode` 改为 `evaluate`，并将 `resume_from_checkpoint` 改为 `best`。

每个 epoch 的完整 checkpoint 使用 `generator_epNNN_origrecX_advrecX_lossX_timeX_sizeX.pt` 命名；`latest.pt` 和 `best.pt` 仍作为稳定的恢复入口。其中 `origrec` 与 `advrec` 是训练期正样本验证 Recall，不是 Precision 或整体 Accuracy。

## 扩展配置

m-DeepCorr 和 DeepCoFFEA 配置可以通过同一个启动器显式运行：

```bash
./run_work2.sh vista_augur/configs/second_workpoint_mdeepcorr_full.json
./run_work2.sh vista_augur/configs/second_workpoint_deepcoffea_full.json
```

这两份配置目前只是后续验证入口，其 batch size、损失权重和负样本数量尚未声明为与 Work1 完全对齐的正式协议。在真正运行前，应先根据对应攻击模型的原始训练与评估代码复核，而不能直接用于论文结论。

## 已删除的旧配置

早期 smoke、probe、旧调参、消融和独立低-FPR配置已经删除。这些用途都可以通过复制主配置并修改 `model_id`、样本上限、step 上限或分支开关完成，不再为每个临时实验维护一份重复文件。
