import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import pathlib
import os
import torch.nn.functional as F
import pickle

# 假设您的模型和数据加载器在以下路径
# 如果不在，请根据您的项目结构修改
from target_model import Deepcoffea
from data_provider.data_loader import DeepCoffeaDataset
from torch.utils.data import DataLoader

def set_seed(seed):
    """
    设置随机种子以确保结果可复现。
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 适用于多GPU情况
    
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")
# ======================================================================================
# 1. 核心功能函数
# ======================================================================================

def partition_single_session(session, delta, win_size, n_wins, tor_len, device):
    """
    划分单条会话，生成窗口、特征结束索引和原始边界。
    """
    win_size_ms = win_size * 1000
    delta_ms = delta * 1000
    offset = win_size_ms - delta_ms

    session_ipd = session[0, :]
    session_size = session[1, :]
    cumulative_time = session_ipd.abs().cumsum(dim=0)
    
    partitioned_data_single = []
    time_indices_single = []
    size_indices_single = []
    original_boundaries_single = []

    for wi in range(int(n_wins)):
        start_time = wi * offset
        end_time = start_time + win_size_ms
        start_idx = torch.searchsorted(cumulative_time, start_time).item()
        end_idx = torch.searchsorted(cumulative_time, end_time).item()
        
        original_boundaries_single.append((start_idx, end_idx))

        window_ipd = session_ipd[start_idx:end_idx]
        window_size = session_size[start_idx:end_idx]

        if len(window_ipd) > 0:
            window_ipd = torch.cat([torch.tensor([0.0]).to(device), window_ipd[1:]])

        len_ipd = len(window_ipd)
        len_size = len(window_size)
        
        time_end_idx = len_ipd - 1 if len_ipd > 0 else -1
        size_end_idx = (len_ipd + len_size - 1) if len_size > 0 else -1
        
        final_tor_len = tor_len * 2
        window_data = torch.cat([window_ipd, window_size])
        if window_data.shape[0] < final_tor_len:
            padding = torch.zeros(final_tor_len - window_data.shape[0], device=device)
            window_data = torch.cat([window_data, padding])
        window_data = window_data[:final_tor_len]

        partitioned_data_single.append(window_data)
        time_indices_single.append(time_end_idx)
        size_indices_single.append(size_end_idx)
    
    partitioned_data = torch.stack(partitioned_data_single, dim=0)
    return partitioned_data, time_indices_single, size_indices_single, original_boundaries_single

def RM_perturb_window(window, time_end_idx, size_end_idx, time_ratio, size_ratio, device):
    """对单个已经切分好的窗口施加随机扰动。"""
    adv_window = window.clone()
    noise = torch.rand_like(window).to(device)
    ratios = torch.zeros_like(window).to(device)
    if time_end_idx >= 0:
        ratios[0 : time_end_idx + 1] = time_ratio
    if size_end_idx > time_end_idx:
        ratios[time_end_idx + 1 : size_end_idx + 1] = size_ratio
    perturbation = noise * torch.abs(window) * ratios
    adv_window = torch.sign(window) * (torch.abs(window) + perturbation)
    return adv_window

def reconstruct_single_session(adv_windows_tensor, original_session, boundaries, time_indices, size_indices):
    """【单条会话版】根据扰动后的窗口和原始边界，还原出完整的对抗会话流。"""
    reconstructed_session = original_session.clone()
    n_wins = adv_windows_tensor.shape[0]

    for j in range(n_wins):
        adv_win = adv_windows_tensor[j]
        s_idx, e_idx = boundaries[j]
        
        if s_idx >= e_idx: continue
            
        t_end_in_win = time_indices[j]
        s_end_in_win = size_indices[j]

        # 从扰动窗口中提取出有效的时间和大小数据（去除padding）
        len_ipd_in_win = t_end_in_win + 1
        len_size_in_win = s_end_in_win - t_end_in_win
        
        adv_win_ipd = adv_win[0 : len_ipd_in_win]
        adv_win_size = adv_win[len_ipd_in_win : len_ipd_in_win + len_size_in_win]
        
        # DeepCoffea中窗口的第一个IPD是伪造的0，所以我们只还原载荷部分
        if len(adv_win_ipd) > 1:
            reconstructed_session[0, s_idx + 1 : s_idx+len(adv_win_ipd[1:])+1] = adv_win_ipd[1:]
        if len(adv_win_size) > 0:
            reconstructed_session[1, s_idx : s_idx+len(adv_win_size)] = adv_win_size
            
    return reconstructed_session

# ======================================================================================
# 2. 主函数 
# ======================================================================================
def main(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 加载模型 ---
    state_dict = torch.load(args.model_path, map_location=device)
    anchor = Deepcoffea.Model(emb_size=args.emb_size, input_size=args.tor_len * 2).to(device)
    pandn = Deepcoffea.Model(emb_size=args.emb_size, input_size=args.exit_len * 2).to(device)
    anchor.load_state_dict(state_dict['anchor_state_dict'])
    pandn.load_state_dict(state_dict['pandn_state_dict'])
    anchor.eval()
    pandn.eval()
    print("Model loaded successfully.")

    # --- 加载数据 ---
    data_filename = f"d{args.delta}_ws{args.win_size}_nw{args.n_wins}_thr{args.threshold}_tl{args.tor_len}_el{args.exit_len}_nt{args.n_test}"
    session_path = os.path.join(args.data_root, "filtered_and_partitioned", f"{data_filename}_test_session.npz")
    win_path = os.path.join(args.data_root, "filtered_and_partitioned", f"{data_filename}_test.npz")
    test_data_session = np.load(session_path, allow_pickle=True)
    test_data_win = np.load(win_path)
    
    indices_to_attack = np.arange(args.num_samples)
    print("Data loaded successfully.")

    # --- 初始化结果存储 ---
    all_original_sims, all_attacked_sims = [], []
    all_sessions_avg_time_ratios, all_sessions_avg_size_ratios = [], []
    total_adv_tor_windows = []
    total_original_tor_windows = []
    total_original_exit_windows = []
    print(f"Starting session-wise attack and evaluation for {args.num_samples} samples...")

    with torch.no_grad():
        for idx in tqdm(indices_to_attack):
            
            # 1. 加载【单条】原始会话流和对应的exit windows
            tor_ipd = torch.tensor(test_data_session['tor_ipds'][idx], dtype=torch.float32).to(device)
            tor_size = torch.tensor(test_data_session['tor_sizes'][idx], dtype=torch.float32).to(device)
            original_session = torch.stack([tor_ipd, tor_size], dim=0)
            original_exit_windows = torch.tensor(test_data_win['test_exit'][:, idx, :], dtype=torch.float32).to(device)

            # 2. 对原始会话进行【第一次拆分】，获取干净窗口和边界信息
            original_tor_windows, time_indices, size_indices, boundaries = \
                partition_single_session(
                    original_session, args.delta, args.win_size, args.n_wins, args.tor_len, device
                )

            # 3. 对每个窗口独立【施加扰动】
            adv_tor_windows_list = []
            for k in range(args.n_wins):
                adv_win = RM_perturb_window(
                    original_tor_windows[k],
                    time_indices[k], size_indices[k],
                    args.time_ratio, args.size_ratio, device
                )
                adv_tor_windows_list.append(adv_win)
            adv_tor_windows_tensor = torch.stack(adv_tor_windows_list, dim=0)

            # --- 评估流程 ---
            # 4. 计算【原始】相似度
            orig_tor_emb = anchor(original_tor_windows)
            orig_exit_emb = pandn(original_exit_windows)
            all_original_sims.append(F.cosine_similarity(orig_tor_emb, orig_exit_emb).mean().item())
            
            total_original_tor_windows.append(original_tor_windows)
            total_original_exit_windows.append(original_exit_windows)
            # 5. 【合并】还原完整的对抗会话流
            reverted_session = reconstruct_single_session(
                adv_tor_windows_tensor, original_session, 
                boundaries, time_indices, size_indices
            )
            
            # 6. 【再次拆分】对扰动后的完整会话流进行重新划分
            repartitioned_adv_windows, _, _, _ = partition_single_session(
                reverted_session, args.delta, args.win_size, args.n_wins, args.tor_len, device
            )

            # 7. 计算【对抗】相似度 (使用再划分后的窗口)
            adv_tor_emb = anchor(repartitioned_adv_windows)
            
            total_adv_tor_windows.append(repartitioned_adv_windows)
            
            all_attacked_sims.append(F.cosine_similarity(adv_tor_emb, orig_exit_emb).mean().item())
            
            # --- 扰动率计算 (按每个窗口的扰动率求平均) ---
            current_session_time_ratios = []
            current_session_size_ratios = []
            total_perturbation_on_windows = adv_tor_windows_tensor - original_tor_windows
            
            for k in range(args.n_wins):
                t_end_idx, s_end_idx = time_indices[k], size_indices[k]
                if t_end_idx < 0 and s_end_idx < 0: continue # 跳过空窗口

                pert_win = total_perturbation_on_windows[k]
                orig_win = original_tor_windows[k]

                pert_time = pert_win[0 : t_end_idx + 1]
                orig_time = orig_win[0 : t_end_idx + 1]
                pert_size = pert_win[t_end_idx + 1 : s_end_idx + 1]
                orig_size = orig_win[t_end_idx + 1 : s_end_idx + 1]
                
                win_time_ratio = torch.linalg.norm(pert_time,ord = 2,dim =-1) / (torch.linalg.norm(orig_time,ord = 2,dim =-1) + 1e-12)
                win_size_ratio = torch.linalg.norm(pert_size,ord = 2,dim =-1) / (torch.linalg.norm(orig_size,ord = 2,dim =-1) + 1e-12)
                print(f"Window {k}: Time Ratio = {win_time_ratio.item():.4f}, Size Ratio = {win_size_ratio.item():.4f}")               
                current_session_time_ratios.append(win_time_ratio.item())
                current_session_size_ratios.append(win_size_ratio.item())

            all_sessions_avg_time_ratios.append(np.nanmean(current_session_time_ratios))
            all_sessions_avg_size_ratios.append(np.nanmean(current_session_size_ratios))

    # --- 整合并输出最终结果 ---
    mean_orig_sim = np.mean(all_original_sims)
    mean_adv_sim = np.mean(all_attacked_sims)
    mean_time_ratio = np.nanmean(all_sessions_avg_time_ratios)
    mean_size_ratio = np.nanmean(all_sessions_avg_size_ratios)

    print("\n--- Final Session-wise Attack Results (with Re-Partitioning & Per-Window Ratios) ---")
    print(f"Processed Samples: {args.num_samples}")
    print(f"Average Similarity BEFORE Attack: {mean_orig_sim:.4f}")
    print(f"Average Similarity AFTER Attack (Re-Partitioned): {mean_adv_sim:.4f}")
    print("-" * 20)
    print(f"Average of Per-Window Time Features L2 Ratio: {mean_time_ratio:.4f}")
    print(f"Average of Per-Window Size Features L2 Ratio: {mean_size_ratio:.4f}")
    result_fpath = pathlib.Path(f'baseline/RM/RMdeepcoffea_advsamples_time{np.mean(mean_time_ratio):.4f}_size{np.mean(mean_size_ratio):.4f}.p')
    with open(result_fpath, "wb") as fp:
        results = {
            "adv_tor_windows": total_adv_tor_windows,
            "original_tor_windows": total_original_tor_windows,
            "original_exit_windows": total_original_exit_windows,
            "time_l2_ratios": np.mean(mean_time_ratio),
            "size_l2_ratios": np.mean(mean_size_ratio),
            "original_sims": np.mean(all_original_sims),
            "attacked_sims": np.mean(all_attacked_sims)
        }
        pickle.dump(results, fp)

# ======================================================================================
# 3. 运行入口
# ======================================================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Final Refactored RM Attack on DeepCoffea.")
    parser.add_argument("--model_path", default='target_model/deepcoffea/deepcoffea_d3_ws5_nw11_thr20_tl500_el800_nt1000_ap1e-01_es64_lr1e-03_mep100000_bs256/best_loss.pth', type=str, help="Path to the pretrained model checkpoint.")
    parser.add_argument("--data_root", type=str, default="target_model/deepcoffea/dataset/CrawlE_Proc/", help="Path to the root directory of the dataset.")
    
    # --- 数据集固有参数 (必须与文件名匹配) ---
    parser.add_argument("--delta", default=3, type=int)
    parser.add_argument("--win_size", default=5, type=int)
    parser.add_argument("--n_wins", default=11, type=int)
    parser.add_argument("--threshold", default=20, type=int)
    parser.add_argument("--tor_len", default=500, type=int)
    parser.add_argument("--exit_len", default=800, type=int)
    parser.add_argument("--n_test", default=1000, type=int)

    # --- 运行和模型结构参数 ---
    parser.add_argument("--emb_size", default=64, type=int)
    parser.add_argument("--num_samples", default=500, type=int, help="Total number of sessions to attack.")
    parser.add_argument("--device", default="cuda:0", type=str, help="GPU device to use (e.g., 'cuda:0' or 'cpu').")

    # --- 扰动超参数 ---
    parser.add_argument("--time_ratio", default=0.20, type=float, help="Perturbation ratio for time features.")
    parser.add_argument("--size_ratio", default=0.20, type=float, help="Perturbation ratio for size features.")
    
    parser.add_argument("--seed", default=2025, type=int, help="Random seed for reproducibility.")

    args = parser.parse_args()
    main(args)