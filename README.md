# VISTA-Augur 第二工作点

本仓库包含 Augur 流量关联攻击防御项目的第二工作点。在保持第一工作点数据集、攻击目标、物理扰动约束和评测协议可对比的前提下，使用基于冻结 Qwen 的轻量多模态重编程模型替换原有扰动生成器。

## 核心实现

- `vista_augur/second_workpoint/`：数据、模型、损失函数、评估与训练代码。
- `vista_augur/configs/`：当前主线和两个后续跨攻击模型预设，详见该目录的 `README.md`。
- `datasets/`：本地 DeepCorr 与 DeepCoFFEA 数据，Git 不跟踪。
- `target_model/`：冻结攻击模型实现及其本地 checkpoint。
- `base_models/`：本地 Hugging Face 基模快照，Git 不跟踪。
- `vista_augur/outputs/`：训练 checkpoint 与日志，Git 不跟踪。

Work2 分别支持 DeepCorr300、两阶段 m-DeepCorr 和 DeepCoFFEA 三个冻结攻击目标。三者使用独立配置和启动脚本，避免结果目录与运行参数混用。

三个目标都以完整流量或完整 session 为一个 dataloader 样本。DeepCorr300 和 m-DeepCorr 在全部生成窗口写回后，把完整流量交给冻结目标；DeepCoFFEA 则在完整 Tor session 写回后，按 Work1 的 IPD 时间边界重新划分 11 个 Tor 窗口，再与预分区的配对 Exit 窗口逐窗计算冻结双塔反馈。

## 环境配置

推荐 Python 3.10 或 3.11。在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

默认骨干模型为 `Qwen/Qwen2.5-1.5B-Instruct`，FP16 下可放入一张 RTX 4090。实验配置使用可迁移的相对路径 `base_models/Qwen2.5-1.5B-Instruct`。请在可联网机器下载快照，再将该 Git 忽略目录复制到离线实验服务器。

```bash
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir base_models/Qwen2.5-1.5B-Instruct
```

## 验证与运行

```bash
PYTHONPATH=vista_augur python -m unittest discover -s vista_augur/tests -v
```

三个目标分别使用以下入口：

```bash
bash run_work2_deepcorr.sh
bash run_work2_mdeepcorr.sh
bash run_work2_deepcoffea.sh
```

三份目标配置分别是 `vista_augur/configs/deepcorr_config.jsonc`、`vista_augur/configs/mdeepcorr_config.jsonc` 和 `vista_augur/configs/deepcoffea_config.jsonc`。每份配置均用分隔线标出日常运行区、主要调参区、目标协议区和资源路径区，并附有中文注释。

启动脚本直接使用 `visible_gpu_devices` 中填写的两张 GPU，不检查显卡占用情况。该配置对齐第一工作点的训练协议：`batch_size=16`、训练期 `drop_last=true`、20 epoch、学习率 `0.01`、每轮乘 `0.8`，损失权重为 `beta/alpha/gamma=1/3/0.9`。

调参时复制该 JSON 并修改 `model_id`，避免覆盖已有 checkpoint。通常只调整 `learning_rate`、`learning_rate_decay`、`beta`、`alpha`、`gamma`、`batch_size`、`train_epochs` 和 `patience`；数据划分及评估字段不应改动。

恢复训练时将 `resume_from_checkpoint` 改为 `latest`；正式评估时将 `run_mode` 改为 `evaluate`、将 `resume_from_checkpoint` 改为 `best`，两者都执行同一条命令：

```bash
bash run_work2_deepcorr.sh
```

`run_mode=evaluate` 默认使用 `final_eval_split=test`，并在 `10^-3`、`10^-4` FPR 下分别为 clean 和 adversarial 分数校准阈值。

## 指标术语

- `original_positive_rate` 与 `adv_positive_rate`：匹配流量对的检测率。
- `attack_success_rate`：原先被检测为匹配、加入扰动后变为未匹配的样本比例。
- `clean_*` 与 `adv_*` 的 Precision、Recall、F1、FPR：以真实匹配对作为正样本、确定性的跨会话错配作为负样本。
- `operating_points`：分别从 clean 和 adversarial 负样本分数中校准论文使用的 `1e-3`、`1e-4` FPR 阈值。正式 DeepCorr 测试使用 1,000 个正样本和每个正样本 199 个负样本。
- DeepCorr 使用第一工作点的概率阈值 `0.1`。
- DeepCoFFEA 使用每条 session 的 11 个真实配对窗口（Tor 500 包、Exit 800 包），训练损失为逐窗口余弦 margin hinge；评估按 Work1 的 9/11 投票得到 session 级判定，不使用全零 Exit 输入。

## 参考资料

- 第一工作点源码：<https://github.com/Ni-Xilin/Augur>
- 本地第一工作点论文：`第一个工作点论文/`
