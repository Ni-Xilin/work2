# target_model 切换说明

## 目标

`work2` 当前仍保留轻量 stub 训练闭环：输入数据、生成器更新和 checkpoint 保存都走 `numpy_stub`。新增的目标模型适配层只替换冻结反馈模型，因此可以先用 stub 快速测试流程，再通过配置切换到最外层 `target_model/` 中的真实代码和权重。

## 配置字段

| 字段 | 默认值 | 作用 |
| --- | --- | --- |
| `target_model_mode` | `stub` | `stub` 使用 `FrozenTargetModelStub`；`torch` 使用最外层真实目标模型。 |
| `target_model` | `Deepcorr300Stub` | 目标模型名称。真实模型可用 `Deepcorr100`、`Deepcorr300`、`Deepcorr700`、`mDeepcorr`、`Deepcoffea`。 |
| `target_model_root` | `target_model` | 最外层目标模型代码根目录。 |
| `target_model_path` | `work2/outputs/target_stub` | 真实权重路径；当 `target_model_mode=torch` 时指向 `.pth` 文件。 |
| `target_model_dropout` | `0.0` | DeepCorr 前向时传入的 dropout，评估/冻结反馈默认使用 `0.0`。 |

## 默认测试路径

默认配置仍然是全 stub：

```bash
python work2/run_train_stub.py --config work2/configs/second_workpoint_stub.json
```

如果只想测试“stub 输入输出 + 真实 DeepCorr300 反馈”，使用：

```bash
N:\anaconda3\envs\myPytorch\python.exe work2/run_train_stub.py --config work2/configs/second_workpoint_stub_deepcorr300_target.json
```

该配置不会读取真实数据集，只会把 stub 生成的 4 通道 Tor flow 映射到 DeepCorr 需要的 8 通道输入位置 `[0, 3, 4, 7]`，用于确认真实目标模型加载和前向输出可用。

## 真实目标模型路径

当前已配置的真实权重位置如下：

| 目标模型 | 配置示例 | 权重 |
| --- | --- | --- |
| DeepCorr300 | `second_workpoint_deepcorr300_real_smoke.json` | `target_model/deepcorr/deepcorr300/tor_199_epoch23_acc0.82dict.pth` |
| DeepCoFFEA | `second_workpoint_deepcoffea_real_smoke.json` | `target_model/deepcoffea/.../best_loss.pth` |

> 注意：真实 target 需要 PyTorch。当前 base Python 没有 `torch`，但 `N:\anaconda3\envs\myPytorch\python.exe` 已验证可用。
