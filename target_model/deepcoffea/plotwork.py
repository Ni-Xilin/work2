import pickle
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse

def get_tprsfprsprs_localthr(corr_matrix, n_wins, vote_thr):
    n_test = corr_matrix.shape[0] // n_wins
    tprs = []
    fprs = []
    prs = []
    for kappa in tqdm(range(0, n_test, 10), ascii=True, ncols=120):
        votes = np.zeros((n_test, n_test), dtype=np.int64)
        for wi in range(0, n_wins):
            corr_matrix_win = corr_matrix[n_test*wi:n_test*(wi + 1), n_test*wi:n_test*(wi + 1)]

            thresholds = []
            for i in range(n_test):
                corr_v = corr_matrix_win[i] # (n_test,)
                corr_v_sorted = np.sort(corr_v)[::-1]
                thresholds.append(corr_v_sorted[kappa])

            for i in range(n_test):
                for j in range(n_test):
                    if corr_matrix_win[i,j] >= thresholds[i]:
                        votes[i,j] += 1

        tp, fp, tn, fn = 0, 0, 0, 0
        for i in range(n_test):
            for j in range(n_test):
                if votes[i,j] >= vote_thr and i == j:
                    tp += 1
                elif votes[i,j] >= vote_thr and i != j:
                    fp += 1
                elif votes[i,j] < vote_thr and i == j:
                    fn += 1
                else:   # votes[i,j] < vote_thr and i != j
                    tn += 1

        if tp + fn == 0:
            tprs.append(0.0)
        else:
            tprs.append(tp / (tp + fn))
        
        if fp + tn == 0:
            fprs.append(0.0)
        else:
            fprs.append(fp / (fp + tn))

        if tp + fp == 0:
            prs.append(0.0)
        else:
            prs.append(tp / (tp + fp))

    return tprs, fprs, prs

def get_tprsfprsprs_globalthr(corr_matrix, n_wins, vote_thr):
    n_test = corr_matrix.shape[0] // n_wins
    tprs = []   # true positive rate, recall, sensitivity
    fprs = []   # false positive rate
    prs = []    # precision, positive predictive value
    for eta in tqdm(np.arange(-0.5, 1, 0.025), ascii=True, ncols=120):
        votes = np.zeros((n_test, n_test), dtype=np.int64)
        for wi in range(0, n_wins):
            corr_matrix_win = corr_matrix[n_test*wi:n_test*(wi + 1), n_test*wi:n_test*(wi + 1)]

            for i in range(n_test):
                for j in range(n_test):
                    if corr_matrix_win[i,j] >= eta:
                        votes[i,j] += 1

        tp, fp, tn, fn = 0, 0, 0, 0
        for i in range(n_test):
            for j in range(n_test):
                if votes[i,j] >= vote_thr and i == j:
                    tp += 1
                elif votes[i,j] >= vote_thr and i != j:
                    fp += 1
                elif votes[i,j] < vote_thr and i == j:
                    fn += 1
                else:   # votes[i,j] < vote_thr and i != j
                    tn += 1

        if tp + fn == 0:
            tprs.append(0.0)
        else:
            tprs.append(tp / (tp + fn))
        
        if fp + tn == 0:
            fprs.append(0.0)
        else:
            fprs.append(fp / (fp + tn))

        if tp + fp == 0:
            prs.append(0.0)
        else:
            prs.append(tp / (tp + fp))

    return tprs, fprs, prs


# best_npz_path = pathlib.Path("datasets/deepcoffea_models/deepcoffea_data230521_d3.0_ws5.0_nw5_thr20_tl300_el500_nt0_ap1e-01_es64_lr1e-03_mep100000_bs256/ep-975_loss0.00199_metrics.npz")
best_npz_path = pathlib.Path("datasets_convert/deepcoffea_PatchTST_Deepcoffea_sl150_pl70_dm512_nh8_pal10_s70/Gcorr_matrix_reverted_sim_0.4813_samples500.npz")
lsetup = "_".join(best_npz_path.parent.parent.name.split("_")[-12:-9])

result_fpath = pathlib.Path("target_model/deepcoffea/results/stats/lthr.p")
result_fpath2 = pathlib.Path("target_model/deepcoffea/results/stats/lthr_d2.p")

if result_fpath.exists():
    with open(result_fpath, "rb") as fp:
        ltprs, lfprs, lprs = pickle.load(fp)
else:
    loaded = np.load(best_npz_path)
    ltprs, lfprs, lprs = get_tprsfprsprs_localthr(loaded['corr_matrix'], 5, 3)
    with open(result_fpath, "wb") as fp:
        pickle.dump((ltprs, lfprs, lprs), fp)
    with open(result_fpath2, "wb") as fp:
        pickle.dump((ltprs, lfprs, lprs), fp)


# In[ ]:


ltprs, lfprs, lprs = {}, {}, {}
with open(result_fpath, "rb") as fp:
    ltprs['d3_ws5_nw11'], lfprs['d3_ws5_nw11'], lprs['d3_ws5_nw11'] = pickle.load(fp)

with open(result_fpath2, "rb") as fp:
    ltprs['d2_ws3_nw7'], lfprs['d2_ws3_nw7'], lprs['d2_ws3_nw7'] = pickle.load(fp)


# In[ ]:


npz_paths = [
    'datasets_convert/deepcoffea_d3_ws5_nw11_thr20_tl500_el800_nt1000_ap1e-01_es64_lr1e-03_mep100000_bs256/deepcoffea_PatchTST_Deepcoffea_sl150_pl70_dm512_nh8_pal10_s70/Gcorr_matrix_reverted_sim_0.4813_samples500.npz'
]

result_fpath = pathlib.Path("target_model/deepcoffea/results/stats/gthr.p")
if result_fpath.exists():
    with open(result_fpath, "rb") as fp:
        gtprs, gfprs, gprs = pickle.load(fp)
else:
    gtprs, gfprs, gprs = {}, {}, {}
    for npz_path in npz_paths:
        npz_path = pathlib.Path(npz_path)
        fields = npz_path.parent.parent.name.split("_")
        n_wins = int(fields[-10].split("nw")[1])
        setup = "_".join(fields[-12:-9])
        
        loaded = np.load(npz_path)
        corr_matrix = loaded['corr_matrix']
        # loss_mean = loaded['loss_mean']

        if n_wins == 5:
            vote_thr = 3
        elif n_wins == 7:
            vote_thr = 4
        elif n_wins == 9:
            vote_thr = 5
        elif n_wins == 11:
            vote_thr = 9    # the number dcf authors used

        tprs, fprs, prs = get_tprsfprsprs_globalthr(corr_matrix, n_wins, vote_thr)
        gtprs[setup] = tprs
        gfprs[setup] = fprs
        gprs[setup] = prs

    with open(result_fpath, "wb") as fp:
        pickle.dump((gtprs, gfprs, gprs), fp)


# In[ ]:


plt.style.use('seaborn-v0_8-paper')
params = {
    'axes.titlesize': 18,
    'axes.labelsize': 20,
    'font.size': 10,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'text.usetex': False,
    'axes.spines.top': False,
    'axes.spines.right': False
}
plt.rcParams.update(params)


# In[ ]:


plt.clf()
fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))

setups = []
for key in gtprs.keys():
    setups.append("_".join(key.split("_")))
setups = sorted(set(setups))

for key, gtprs_data in gtprs.items():
    gfprs_data = gfprs[key]
    ltprs_data = ltprs[key]
    lfprs_data = lfprs[key]
    
    setupi = setups.index(key)
    if setupi == 0:
        linestyle = "solid"
        label = "setup-1"
    elif setupi == 1:
        linestyle = "dashed"
        label = "setup-2"
    else:
        raise ValueError(f"setupi: {setupi} not supported now.")
    
    ax.plot(gfprs_data, gtprs_data, linewidth=1.5, linestyle=linestyle, label=f"{label}_g",marker='o')
    ax.plot(lfprs_data, ltprs_data, linewidth=1.5, linestyle=linestyle, label=f"{label}_l",marker='o')

ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.legend(loc="lower right", fontsize=20)

ax.set_xscale('log')
ax.set_xlim(0.00002,1)

fig.tight_layout()
fig.savefig(f"target_model/deepcoffea/results/stats/dcf_eval_roc.pdf")
fig.savefig(f"target_model/deepcoffea/results/stats/dcf_eval_roc.png")

plt.clf()
fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
for key, gtprs_data in gtprs.items():
    gprs_data = gprs[key]
    ltprs_data = ltprs[key]
    lprs_data = lprs[key]
    
    setupi = setups.index(key)
    if setupi == 0:
        linestyle = "solid"
        label = "setup-1"
    elif setupi == 1:
        linestyle = "dashed"
        label = "setup-2"
    else:
        raise ValueError(f"setupi: {setupi} not supported now.")
    
    ax.plot(gtprs_data, gprs_data, linewidth=1.5, linestyle=linestyle, label=f"{label}_g",marker='o')
    # ax.plot(ltprs_data, lprs_data, linewidth=1.5, linestyle=linestyle, label=f"{label}_l",marker='o')
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.legend(loc="upper right", fontsize=20)
fig.tight_layout()
fig.savefig(f"target_model/deepcoffea/results/stats/dcf_eval_pr.pdf")
fig.savefig(f"target_model/deepcoffea/results/stats/dcf_eval_pr.png")
