# target_model 接入说明

## 这份文档回答什么

第二工作点现在已经不再保留 stub target model。`work2/second_workpoint` 里的目标反馈模块只做一件事：

- 根据配置加载最外层 `target_model/` 目录中的真实 DeepCorr / DeepCoFFEA 代码与权重
- 在训练中保持参数冻结
- 允许梯度从 target logits 对上游 generator 输出回传

## 当前支持的目标模型

| `target_model` | 实际加载对象 | 典型权重路径 |
| --- | --- | --- |
| `Deepcorr100` | `target_model/Deepcorr100.py` | `target_model/deepcorr/deepcorr100/...pth` |
| `Deepcorr300` | `target_model/Deepcorr300.py` | `target_model/deepcorr/deepcorr300/tor_199_epoch23_acc0.82dict.pth` |
| `Deepcorr700` / `mDeepcorr` | `target_model/Deepcorr700.py` | `target_model/deepcorr/deepcorr700/...pth` |
| `Deepcoffea` | `target_model/Deepcoffea.py` | `target_model/deepcoffea/.../best_loss.pth` |

## 关键配置字段

| 字段 | 作用 | 当前要求 |
| --- | --- | --- |
| `target_model` | 选择目标模型族 | 必填，且必须是受支持的真实模型名 |
| `target_model_root` | 目标模型代码根目录 | 默认是 `target_model` |
| `target_model_path` | 目标模型权重路径 | 可留空走默认权重，也可显式指定 |
| `target_model_dropout` | DeepCorr 前向时的 dropout | 默认建议 `0.0` |
| `target_device` | 目标模型所在设备 | 默认与 trainable 模块同卡 |

## 数据形状是怎么对齐的

### DeepCorr 路径

- generator 侧默认生成的是 Tor 视角 `4` 通道流量
- 如果 target 需要 `8` 通道输入，适配层会把这 `4` 个通道写回到 `[0, 3, 4, 7]`
- 如果 generator 侧已经提供了 `target_full_flow`，则优先在那份真实 `8` 通道流量上写回 future 扰动

### DeepCoFFEA 路径

- 只使用 Tor 侧前两路：`ipd` 和 `size`
- 会把长度裁剪或补零到 `deepcoffea_tor_len`
- exit 侧输入在当前最小链路里仍然使用零张量占位，目的是先验证 generator -> target -> loss 这条真实反馈链

## 为什么 target 可以冻结但仍然参与训练

冻结 target 的意思只是：

- `requires_grad=False`
- 不更新 target 参数

它**不等于**：

- 用 `torch.no_grad()` 包住整个 target forward

当前实现选择的是前者。这样做的结果是：

- target 自己不学习
- 但 target logits 对 `adv_flow` 的梯度仍然存在
- generator 侧可以继续根据这份反馈学习如何生成 future 扰动

## 当前推荐的配置方式

### DeepCorr300 smoke

配置文件：

- `work2/configs/second_workpoint_deepcorr300_torch_real_smoke.json`

### DeepCoFFEA smoke

配置文件：

- `work2/configs/second_workpoint_deepcoffea_real_smoke.json`

## 一个常见误解

> “target_model 既然冻结，那它只是评估器，不属于训练链。”

这个理解不准确。当前第二工作点里，冻结 target 不是离线评估器，而是**参与 loss 构造的可微反馈模块**。它不更新参数，但它定义了 generator 被优化的方向。
