# `plot/` 目录文件与前置输入说明

本文档用中文说明 `plot/` 下每个文件的作用、图表含义和运行前必须准备的文件。命令均在仓库根目录执行。

## 总体流程

训练后，`run_train.py --evaluate` 会得到原始 `evaluation_summary.json` 和 `evaluation_scores.npz`。绘图器通常不直接读取原始 summary，而是先运行 `data.evaluate`，生成标准化的 `*.summary.json`、配套 `*.curves.npz` 和 `index.json`。

## 文件说明

| 文件 | 作用 | 直接前置文件 | 默认输出 |
| --- | --- | --- | --- |
| `__init__.py` | 声明 Python 包，不绘图 | 无 | 无 |
| `common.py` | 统一 Matplotlib 样式、读取 summary/曲线、标签、颜色和保存逻辑 | 无独立输入 | 无 |
| `plot_roc.py` | 按 `target` 分面绘制 ROC；横轴 FPR（对数），纵轴 TPR，每个 condition 一条线 | `*.summary.json` + 其 `curves` 指向的 `*.curves.npz` | `plot/results/roc.pdf` |
| `plot_operating_points.py` | 指定低 FPR 的 Precision、Recall、F1 分组柱状图 | 含 `conditions.*.operating_points.<fpr>` 的 summary | `plot/results/operating_points.pdf` |
| `plot_score_matrix.py` | 绘制 clean/defended 相关性分数方阵热图，共享颜色范围 | 含二维方阵的 NPZ；键由 `--keys` 指定 | `plot/results/score_matrices.pdf` |
| `plot_overhead.py` | 绘制时间 L2、大小 L2、相对延迟开销、相对大小开销 | 含 `overhead` 节点的 summary | `plot/results/overhead.pdf` |
| `plot_training.py` | 三联图：评估 loss、攻击正例率、时间/大小相对 L2 随 epoch 的变化 | `train_summary.json` | `plot/results/training_dynamics.pdf` |
| `plot_efficiency.py` | 绘制选定效率指标的均值柱、P95 菱形和 P99 叉号 | 效率 JSON，指标下含 `mean/p95/p99` | `plot/results/efficiency.pdf` |
| `plot_ablation.py` | ROC 的消融包装器，固定标题为 `Ablation study` | 消融 summary + curve NPZ | `plot/results/ablation.pdf` |
| `plot_transferability.py` | ROC 的迁移/自适应攻击包装器 | 对应 summary + curve NPZ | `plot/results/transferability.pdf` |
| `plot_all.py` | 读取 figure manifest，依次启动上述绘图模块；自身不画图 | `--manifest` 指定的 JSON 及其全部输入 | 由清单决定 |
| `figures.example.json` | 批量绘图示例清单，目前只声明主 ROC 和低 FPR 图 | 清单引用的 summary 文件 | 由清单决定 |

## 标准评估文件

`evaluation_scores.npz` 是 `data.evaluate` 的输入，通常包含 `clean_positive_scores`、`clean_negative_scores`、`adv_positive_scores` 和 `adv_negative_scores` 四个一维数组。它不是 ROC 曲线文件，也不是方阵。

标准 `*.summary.json` 至少应包含 `name`、`target`、`method`、`conditions` 和 `curves`。`conditions.<condition>.operating_points.<fpr>` 中应有 `precision`、`recall`、`f1`；ROC 所需曲线 NPZ 至少有 `<condition>__fpr` 和 `<condition>__tpr`。曲线 NPZ 通常和 summary 位于同一目录。

训练动态要求 `train_summary.json` 的每个 epoch 含 `eval.loss`、`eval.adv_positive_rate`、`eval.time_ratio` 和 `eval.size_ratio`。DeepCoFFEA 路径会把 `eval` 写成 `null`，所以当前 `plot_training.py` 无法直接绘制该类训练摘要。

效率图的 JSON 通常由 `data.benchmark` 生成 NPZ 后，再由 `data.evaluate_efficiency` 汇总得到；常见指标是 `latency_ms`、`throughput_per_second`、`cpu_percent` 和 `peak_ram_mb`。

## 开销和方阵的额外输入

开销图的评估清单必须指定 `overhead_artifact`。该 NPZ 必须包含形状为 `(..., channels, length)` 的 `original` 和 `adversarial`，可选 `valid_mask`，并指定 `time_channels`、`size_channels` 及单位换算。当前训练器不会自动导出这些 packet-level 数组。

方阵热图需要目标模型全配对评估导出的二维方阵，例如 `clean_matrix` 和 `adv_matrix`。行列顺序必须一致；一维 score 数组无法恢复方阵。当前仓库没有通用方阵导出入口。

## 生成顺序

1. 训练主实验、消融、迁移和自适应攻击设置，保存 `train_summary.json`。
2. 评估 checkpoint，获得 `evaluation_scores.npz`。
3. 复制 `data/manifest.example.json`，运行 `python -m data.evaluate --manifest ... --output-dir data/results`。
4. 运行 `data.validate_results`，再绘制低 FPR 图。
5. 另行准备 packet-level 开销数组、相关性方阵和 benchmark 测量。
6. 生成效率 JSON，扩展 `plot/figures.example.json`，最后运行 `plot_all.py`。

`1e-4` FPR 至少需要 10,000 个负样本才能经验上分辨。当前 QA summary 只有 `fpr_1e-01` 和 `fpr_1e-02`，不能直接用于默认的 `fpr_1e-04`。最终图片不能替代 summary、curve、实验配置、checkpoint、样本数量和阈值信息。
