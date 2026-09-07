# Work2 配置文件说明

本目录分别保存 DeepCorr300、m-DeepCorr 和 DeepCoFFEA 三个 Work2 目标的独立配置。三个启动脚本各自绑定一份配置，避免误用目标权重或覆盖其他实验目录。

## 配置一览

| 文件 | 用途 | 当前定位 |
| --- | --- | --- |
| `deepcorr_config.jsonc` | 使用冻结 DeepCorr300 反馈训练 Work2 生成器，并按 Work1 的数据划分和评估协议执行最终测试 | 已完成的 DeepCorr 主线 |
| `mdeepcorr_config.jsonc` | 使用冻结 DeepCorr100 筛选和冻结 DeepCorr700 复判组成的两阶段目标反馈 | m-DeepCorr 主线 |
| `deepcoffea_config.jsonc` | 使用冻结 Anchor/PandN 双塔和真实 Tor/Exit 配对反馈 | DeepCoFFEA 主线 |

DeepCorr300 和 mDeepCorr 使用相互独立的后续输出位置：

- DeepCorr300：`vista_augur/outputs/checkpoint_deepcorr300/`
- mDeepCorr（训练反馈模型为 DeepCorr700）：`vista_augur/outputs/checkpoint_deepcorr700/`

这些配置只决定后续运行的保存位置，不会移动、重命名或删除已经生成的 checkpoint 目录。
如果需要恢复旧目录中的训练，`resume_from_checkpoint` 必须填写旧 `latest.pt` 的完整路径；简写 `latest` 会从新的目标目录查找。

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

DeepCoFFEA 已采用与 Work1 一致的 session 级训练协议：每个 dataloader 样本是一条完整 Tor session，Work2 对该 session 的全部生成窗口写回扰动后，再按 IPD 时间边界重新划分为 11 个 DeepCoFFEA Tor 窗口，并与 `*_train.npz` / `*_test.npz` 中预分区的配对 Exit 窗口逐窗计算余弦损失。

- `deepcoffea_delta_seconds=3.0`、`deepcoffea_window_seconds=5.0`、`deepcoffea_n_windows=11`：控制 Work1 的重叠时间窗口协议。
- `deepcoffea_vote_threshold=9`：评估时按 11 个窗口中至少 9 个通过阈值来判定 session 匹配。
- `deepcoffea_similarity_margin=-0.5`、`beta=4.0`：对应 Work1 的负标签 `CosineEmbeddingLoss`。实现上等价为 `max(cosine_similarity - margin, 0)`。
- `learning_rate=0.01`、`lradj=type4`：前两次 epoch 调度保持初始学习率，之后按 `0.7` 指数衰减。
- `batch_size=1`、`gradient_accumulation_steps=16`：物理 batch 降为一条 session，有效 batch size 保持为 16，避免多条长 session 同时展开挤满 Qwen 所在显卡。
- `deepcoffea_generator_window_batch_size=16`、`backbone_activation_checkpointing=true`：每次只向 Qwen 发送 16 个生成窗口，并在反向传播时重算冻结 Qwen 的激活，控制超长 session 的峰值显存。
- 输出目录为 `vista_augur/outputs/checkpoint_deepcoffea/`，不会与 DeepCorr300 或 mDeepCorr checkpoint 混放。

DeepCoFFEA 的 `max_train_steps=21` 对齐 Work1 中 `i > 19` 才停止的实际行为，即每个 epoch 处理 21 个随机 batch。设为 `0` 才会完整遍历 13097 条训练 session，但不建议在首次正式运行时直接启用。

DeepCoFFEA 训练阶段同样对齐 Work1：每个 epoch 只汇总训练集上的余弦 hinge、时间 L2、大小 L2 和总损失，不调用 978 条测试 session 的完整评估，也不启用基于验证集的早停。每个 epoch 仍保存 `latest.pt` 和带 `ep/cos/loss/time/size` 的 checkpoint；`best.pt` 按训练总损失更新。需要完整测试时，将 `run_mode` 改为 `evaluate` 后单独运行启动脚本。

## 历史配置整理

早期 smoke、probe、旧调参、消融和独立低-FPR配置已经删除。这些用途都可以通过复制主配置并修改 `model_id`、样本上限、step 上限或分支开关完成，不再为每个临时实验维护一份重复文件。
