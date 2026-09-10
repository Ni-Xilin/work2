import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
import argparse
from tqdm import tqdm
import pathlib

# 导入DeepCoffea模型定义
from target_model.Deepcoffea import Model as DeepCoffeaModel

# ------------------------------------------------------------
# 1. 数据分区函数 
# ------------------------------------------------------------
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
# ------------------------------------------------------------
# 2. PGD for DeepCoffea
# ------------------------------------------------------------
def deepcoffea_PGD_attack(anchor_model, pandn_model, original_tor_window, original_exit_window,
                                 time_end_idx, size_end_idx, device,
                                 max_ratio_time, max_ratio_size,
                                 alpha_tensor,
                                 num_iter
                                 ):
    """
    最终版PGD攻击，保证扰动增加特征绝对值，同时保留原始符号。
    """
    anchor_model.eval()
    pandn_model.eval()

    adv_window = original_tor_window.clone().detach().unsqueeze(0).to(device)
    original_tor_window_dev = original_tor_window.clone().detach().unsqueeze(0).to(device)
    exit_window_dev = original_exit_window.clone().detach().unsqueeze(0).to(device)
    original_sign = torch.sign(original_tor_window_dev) # 预先计算原始样本的符号

    with torch.no_grad():
        original_time_features = original_tor_window_dev[:, 0:time_end_idx+1] if time_end_idx >= 0 else torch.tensor([], device=device)
        original_size_features = original_tor_window_dev[:, time_end_idx+1:size_end_idx+1] if size_end_idx > time_end_idx else torch.tensor([], device=device)
        original_norm_time = torch.linalg.norm(original_time_features,ord=2,dim=-1)
        original_norm_size = torch.linalg.norm(original_size_features,ord=2,dim=-1)
        epsilon_time_abs = original_norm_time * max_ratio_time
        epsilon_size_abs = original_norm_size * max_ratio_size
        exit_embedding = pandn_model(exit_window_dev)

    # 随机初始化 (增加绝对值)
    initial_delta = (torch.rand_like(adv_window) * 2 - 1) * original_sign * 0.001
    adv_sample = adv_window + initial_delta

    for i in range(num_iter):
        adv_sample.requires_grad = True
        anchor_model.zero_grad()
        adv_embedding = anchor_model(adv_sample)
        loss = F.cosine_similarity(adv_embedding.unsqueeze(0), exit_embedding.unsqueeze(0)).mean()
        loss.backward()

        if adv_sample.grad is None: break
        
        full_grad = adv_sample.grad.data
        
        # --- 核心修改：保证扰动增加特征绝对值 ---
        step = -alpha_tensor * full_grad
        valid_update_mask = (torch.sign(step) == original_sign).float()
        filtered_step = step * valid_update_mask
        adv_sample = adv_sample.data + filtered_step

        # --- 分离式投影 (提前终止) ---
        total_perturbation = adv_sample - original_tor_window_dev
        
        pert_time = total_perturbation[:, 0:time_end_idx+1]
        norm_time = torch.linalg.norm(pert_time,ord=2,dim=-1)
        
        pert_size = total_perturbation[:, time_end_idx+1:size_end_idx+1]
        norm_size = torch.linalg.norm(pert_size,ord=2,dim=-1)
        if ((norm_time /original_norm_time).mean()> max_ratio_time) or \
           ((norm_size /original_norm_size).mean()> max_ratio_size):

            if (norm_time/original_norm_time).mean()>max_ratio_time:
                ratio = norm_time/original_norm_time
                scaling_factor = torch.ones_like(ratio)
                exceed = ratio > max_ratio_time
                scaling_factor[exceed] = max_ratio_time / ratio[exceed]
                total_perturbation[:, 0:time_end_idx+1] *= scaling_factor

            if (norm_size /original_norm_size).mean()> max_ratio_size:
                ratio = norm_size/original_norm_size
                scaling_factor = torch.ones_like(ratio)
                exceed = ratio > max_ratio_size
                scaling_factor[exceed] = max_ratio_size / ratio[exceed]
                total_perturbation[:, time_end_idx+1:size_end_idx+1] *= scaling_factor
                
            adv_sample = original_tor_window_dev + total_perturbation
            break
        
        adv_sample = original_tor_window_dev + total_perturbation
    print(f"Stopping at iteration {i+1} due to L2 constraint.")
    print(f"Time perturbation ratio: {(torch.linalg.norm(pert_time,ord=2,dim=-1) / original_norm_time).mean().item():.4f}, Size perturbation ratio: {(torch.linalg.norm(pert_size,ord=2,dim=-1) / original_norm_size).mean().item():.4f}")
    return adv_sample.squeeze(0).detach()


# ------------------------------------------------------------
# 3. 主函数
# ------------------------------------------------------------
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 加载DeepCoffea模型 ---
    print("正在加载 DeepCoffea 模型...")
    ckpt_path = pathlib.Path(args.target_model_path).resolve()
    fields = ckpt_path.name.split("_")
    emb_size = int(fields[-4].split("es")[-1])
    tor_len = int(fields[-8].split("tl")[-1])
    exit_len = int(fields[-7].split("el")[-1])

    anchor_model = DeepCoffeaModel(emb_size=emb_size, input_size=tor_len * 2).to(device)
    pandn_model = DeepCoffeaModel(emb_size=emb_size, input_size=exit_len * 2).to(device)

    state_dict = torch.load(os.path.join(args.target_model_path, 'best_loss.pth'), map_location=device)
    anchor_model.load_state_dict(state_dict['anchor_state_dict'])
    pandn_model.load_state_dict(state_dict['pandn_state_dict'])
    anchor_model.eval()
    pandn_model.eval()
    print("模型加载成功。")

    # --- 加载数据集 ---
    print("正在加载数据集...")
    data_filename = f"d{args.delta}_ws{args.win_size}_nw{args.n_wins}_thr{args.threshold}_tl{args.tor_len}_el{args.exit_len}_nt{args.n_test}"
    session_path = os.path.join(args.data_path, "filtered_and_partitioned", f"{data_filename}_test_session.npz")
    win_path = os.path.join(args.data_path, "filtered_and_partitioned", f"{data_filename}_test.npz")
    test_data_session = np.load(session_path, allow_pickle=True)
    test_data_win = np.load(win_path)
    
    indices_to_attack = np.arange(args.num_samples)

    # --- 执行攻击和评估 ---
    all_original_sims, all_attacked_sims = [], []
    time_l2_ratios = []
    size_l2_ratios = []
    total_adv_tor_windows, total_original_tor_windows, total_original_exit_windows = [], [], []
    total_repartitioned_adv_windows = []
    print(f"开始对 {args.num_samples} 个样本进行 Custom PGD 攻击...")
    for idx in tqdm(indices_to_attack):
        tor_ipd = torch.tensor(test_data_session['tor_ipds'][idx], dtype=torch.float32).to(device)
        tor_size = torch.tensor(test_data_session['tor_sizes'][idx], dtype=torch.float32).to(device)
        original_session = torch.stack([tor_ipd, tor_size], dim=0)
        original_exit_windows = torch.tensor(test_data_win['test_exit'][:,idx,:], dtype=torch.float32).to(device)

        original_tor_windows, time_indices, size_indices,boundaries = partition_single_session(
            original_session, args.delta, args.win_size, args.n_wins, args.tor_len, device)
        
        
        adv_tor_windows_list = []
        for i in range(args.n_wins):
            tor_win = original_tor_windows[i]
            exit_win = original_exit_windows[i]
            t_end_idx = time_indices[i]
            s_end_idx = size_indices[i]
            
            # 为当前窗口创建alpha_tensor
            alpha_tensor = torch.zeros_like(tor_win).unsqueeze(0).to(device)
            if t_end_idx >= 0: alpha_tensor[:, 0:t_end_idx+1] = args.alpha_time
            if s_end_idx > t_end_idx: alpha_tensor[:, t_end_idx+1:s_end_idx+1] = args.alpha_size

            adv_win = deepcoffea_PGD_attack(
                anchor_model, pandn_model, tor_win, exit_win,
                t_end_idx, s_end_idx, device,
                args.max_ratio_time, args.max_ratio_size,
                alpha_tensor, args.num_iter
            )
            adv_tor_windows_list.append(adv_win)
        adv_tor_windows = torch.stack(adv_tor_windows_list, dim=0)
        reverted_session = reconstruct_single_session(
                    adv_tor_windows, original_session, 
                    boundaries, time_indices, size_indices
            )
        repartitioned_adv_windows, _, _, _ = partition_single_session(
                reverted_session, args.delta, args.win_size, args.n_wins, args.tor_len, device
            )
        with torch.no_grad():
            orig_tor_emb = anchor_model(original_tor_windows)
            orig_exit_emb = pandn_model(original_exit_windows)
            all_original_sims.append(F.cosine_similarity(orig_tor_emb, orig_exit_emb).mean().item())

            adv_tor_emb =  anchor_model(repartitioned_adv_windows)
            all_attacked_sims.append(F.cosine_similarity(adv_tor_emb, orig_exit_emb).mean().item())

            total_perturbation = adv_tor_windows - original_tor_windows
            session_time_ratios, session_size_ratios = [], []
            for i in range(args.n_wins):
                t_end_idx, s_end_idx = time_indices[i], size_indices[i]
                pert_time = total_perturbation[i, 0:t_end_idx+1] if t_end_idx >= 0 else torch.tensor([], device=device)
                orig_time = original_tor_windows[i, 0:t_end_idx+1] if t_end_idx >= 0 else torch.tensor([], device=device)
                pert_size = total_perturbation[i, t_end_idx+1:s_end_idx+1] if s_end_idx > t_end_idx else torch.tensor([], device=device)
                orig_size = original_tor_windows[i, t_end_idx+1:s_end_idx+1] if s_end_idx > t_end_idx else torch.tensor([], device=device)
                
                session_time_ratios.append(((torch.linalg.norm(pert_time,ord=2,dim=-1) / (torch.linalg.norm(orig_time,ord=2,dim=-1) + 1e-12))).mean().item())
                session_size_ratios.append(((torch.linalg.norm(pert_size,ord=2,dim=-1) / (torch.linalg.norm(orig_size,ord=2,dim=-1) + 1e-12))).mean().item())
            
        time_l2_ratios.append(np.nanmean(session_time_ratios))
        size_l2_ratios.append(np.nanmean(session_size_ratios))
        total_repartitioned_adv_windows.append(repartitioned_adv_windows.cpu())
        total_original_tor_windows.append(original_tor_windows.cpu())
        total_original_exit_windows.append(original_exit_windows.cpu())
        

    # --- 打印最终结果 ---
    print("\n--- Custom PGD Attack on DeepCoffea: Final Results ---")
    print(f"Attacked Samples: {args.num_samples}")
    print(f"Average Similarity BEFORE Attack: {np.mean(all_original_sims):.4f}")
    print(f"Average Similarity AFTER Attack:  {np.mean(all_attacked_sims):.4f}")
    print("-" * 20)
    print(f" Average L2 Perturbation Ratio (Time): {np.mean(time_l2_ratios):.4f}")
    print(f" Average L2 Perturbation Ratio (Size): {np.mean(size_l2_ratios):.4f}")
    result_fpath = pathlib.Path(f'baseline/PGD/PGDdeepcoffea_advsamples_time{np.mean(time_l2_ratios):.4f}_size{np.mean(size_l2_ratios):.4f}.p')
    with open(result_fpath, "wb") as fp:
        results = {
            "adv_tor_windows": total_repartitioned_adv_windows,
            "original_tor_windows": total_original_tor_windows,
            "original_exit_windows": total_original_exit_windows,
            "time_l2_ratios": np.mean(time_l2_ratios),
            "size_l2_ratios": np.mean(size_l2_ratios),
            "original_sims": np.mean(all_original_sims),
            "attacked_sims": np.mean(all_attacked_sims)
        }
        pickle.dump(results, fp)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Custom PGD attack on DeepCoffea")
    parser.add_argument("--target_model_path", type=str, default="target_model/deepcoffea/deepcoffea_d3_ws5_nw11_thr20_tl500_el800_nt1000_ap1e-01_es64_lr1e-03_mep100000_bs256/", help="Path to DeepCoffea model folder.")
    parser.add_argument("--data_path", type=str, default="target_model/deepcoffea/dataset/CrawlE_Proc/", help="Path to dataset root directory.")
    parser.add_argument("--device", type=str, default="cuda:1", help="Computation device.")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to attack.")

    # Data Parameters
    parser.add_argument("--delta", default=3, type=int)
    parser.add_argument("--win_size", default=5, type=int)
    parser.add_argument("--n_wins", default=11, type=int)
    parser.add_argument("--threshold", default=20, type=int)
    parser.add_argument("--tor_len", default=500, type=int)
    parser.add_argument("--exit_len", default=800, type=int)
    parser.add_argument("--n_test", default=1000, type=int)

    # Attack Hyperparameters
    parser.add_argument("--num_iter", type=int, default=500, help="PGD iterations.")
    parser.add_argument("--max_ratio_time", type=float, default=0.15, help="Max L2 ratio for time features.")
    parser.add_argument("--max_ratio_size", type=float, default=0.15, help="Max L2 ratio for size features.")
    parser.add_argument("--alpha_time", type=float, default=500, help="Step size for time features.")
    parser.add_argument("--alpha_size", type=float, default=0.5, help="Step size for size features.")
    
    args = parser.parse_args()
    main(args)