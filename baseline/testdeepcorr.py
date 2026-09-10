from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pickle
import pathlib
import numpy as np
from scipy.interpolate import interp1d
# fpr: False Positive Rate
# tpr: True Positive Rate
# thresholds: roc_curve使用的阈值
result_fpath1 = pathlib.Path("target_model/deepcorr/deepcorr100/test_index100_result.p")
result_fpath2 = pathlib.Path("target_model/deepcorr/deepcorr300/RMtest_index300_result.p")
result_fpath3 = pathlib.Path("target_model/deepcorr/deepcorr300/Gtest_index300_result.p")

with open(result_fpath1, "rb") as fp:
    all_outputs,all_labels = pickle.load(fp)
with open(result_fpath2, "rb") as fp:
    all_outputs_RM,all_labels_RM,_,_ = pickle.load(fp)
with open(result_fpath3, "rb") as fp:
    all_outputs_G,all_labels_G,_,_ = pickle.load(fp)

threshold = 0.3
label = 1

positive_labels = all_labels[all_labels == label]
positive_labels_RM = all_labels_RM[all_labels_RM == label]
positive_labels_G = all_labels_G[all_labels_G == label]

positive_outputs = all_outputs[all_labels == label]
positive_outputs_RM = all_outputs_RM[all_labels_RM == label]
positive_outputs_G = all_outputs_G[all_labels_G == label]


positive_outputs_above_0_1 = positive_outputs[positive_outputs > threshold]
positive_outputs_RM_above_0_1 = positive_outputs_RM[positive_outputs_RM > threshold]
positive_outputs_G_above_0_1 = positive_outputs_G[positive_outputs_G > threshold]

# 打印结果
if(label == 1):
    print(f"Labels 等于 {label} threshhold为 {threshold} 的正确率为:", len(positive_outputs_above_0_1)/len(positive_labels))
    print(f"Labels 等于 {label} threshhold为 {threshold} 的正确率为:", len(positive_outputs_RM_above_0_1)/len(positive_labels_RM))
    print(f"Labels 等于 {label} threshhold为 {threshold} 的正确率为:", len(positive_outputs_G_above_0_1)/len(positive_labels_G))
else:
    print(f"Labels 等于 {label} threshhold为 {threshold} 的正确率为:",1- len(positive_outputs_above_0_1)/len(positive_labels))
    print(f"Labels 等于 {label} threshhold为 {threshold} 的正确率为:",1- len(positive_outputs_RM_above_0_1)/len(positive_labels_RM))
    print(f"Labels 等于 {label} threshhold为 {threshold} 的正确率为:",1-len(positive_outputs_G_above_0_1)/len(positive_labels_G))