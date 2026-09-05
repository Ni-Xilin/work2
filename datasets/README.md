# 数据集目录

本目录集中存放项目运行所需的数据集，不随模型代码放在 `Generator_Trainer/` 或 `target_model/` 内。

## 当前目录结构

```text
datasets/
├── deepcorr/
│   ├── 8872_tordata300.pickle
│   ├── ...
│   ├── val_index300.pickle
│   └── test_index300.pickle
└── deepcoffea/
    └── CrawlE_Proc/
        └── filtered_and_partitioned/
            ├── d3_ws5_nw11_thr20_tl500_el800_nt1000_train_session.npz
            └── d3_ws5_nw11_thr20_tl500_el800_nt1000_test_session.npz
```

第二工作点配置中的 `data_path` 应从项目根目录解析：

- DeepCorr：`datasets/deepcorr`
- DeepCoFFEA：`datasets/deepcoffea/CrawlE_Proc/filtered_and_partitioned`

数据集文件通常体积较大，已由根目录 `.gitignore` 排除，不应提交到 Git。
