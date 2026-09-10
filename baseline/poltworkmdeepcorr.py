from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pickle
import pathlib
import numpy as np
from scipy.interpolate import interp1d
# fpr: False Positive Rate
# tpr: True Positive Rate
# thresholds: roc_curve使用的阈值
result_fpath = pathlib.Path("baseline/AJSMA/mdeepcorr/AJSMAmdeepcorr_threshDC100_0.0100.p")
output_filename = "target_model/mdeepcorr/results/AJSMAroc_curvelog0.0100.png"
with open(result_fpath, "rb") as fp:
    data = pickle.load(fp)
all_outputs = data["all_outputs"]
all_labels = data["all_labels"]
fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)
# fpr, unique_indices = np.unique(fpr, return_index=True)
# tpr = tpr[unique_indices]
# fpr_smooth = np.logspace(-4, -1, 500)  # 在对数尺度上生成平滑的 fpr 值
# tpr_smooth = interp1d(fpr, tpr, kind='linear', fill_value=(0, 1), bounds_error=False)(fpr_smooth)
# print(f"thresholds: {thresholds}")
# 计算AUC (Area Under the Curve)
roc_auc = auc(fpr, tpr)

# 3. 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.xscale('log')
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # 绘制对角线
plt.xlim([0.0002, 1])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

plt.grid(True)
plt.savefig(output_filename)
print(f"ROC curve plot saved to {output_filename}")
plt.show()

print(f"AUC: {roc_auc}")