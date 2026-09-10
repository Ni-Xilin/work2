import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pathlib
import argparse
import os

# Import the DeepCoffea model definition
from target_model.Deepcoffea import Model as DeepCoffeaModel

# ------------------------------------------------------------
# 1. Data Partitioning Function (from AJSMAdeepcoffea.py)
# ------------------------------------------------------------
# def partition_sessions_with_end_indices(sessions, delta, win_size, n_wins, tor_len, device):
#     """
#     Partitions a single session into windows according to DeepCoffea's methodology
#     and returns the end indices for time and size features within each window.
#     """
#     win_size_ms = win_size * 1000
#     delta_ms = delta * 1000
#     offset = win_size_ms - delta_ms

#     session_ipd = sessions[0, :]
#     session_size = sessions[1, :]
#     cumulative_time = session_ipd.abs().cumsum(dim=0)
    
#     partitioned_data_single = []
#     time_indices_single = []
#     size_indices_single = []

#     for wi in range(int(n_wins)):
#         start_time = wi * offset
#         end_time = start_time + win_size_ms

#         start_idx = torch.searchsorted(cumulative_time, start_time).item()
#         end_idx = torch.searchsorted(cumulative_time, end_time).item()

#         window_ipd = session_ipd[start_idx:end_idx]
#         window_size = session_size[start_idx:end_idx]

#         if len(window_ipd) > 0:
#             window_ipd = torch.cat([torch.tensor([0.0]).to(device), window_ipd[1:]])

#         len_ipd = len(window_ipd)
#         len_size = len(window_size)
        
#         # The window is formed by concatenating IPD and size data
#         # Record the last valid index for each feature type
#         time_end_idx = len_ipd - 1 if len_ipd > 0 else -1
#         size_end_idx = (len_ipd + len_size - 1) if len_size > 0 else -1
        
#         final_tor_len = tor_len * 2
        
#         window_data = torch.cat([window_ipd, window_size])

#         # Pad if necessary
#         if window_data.shape[0] < final_tor_len:
#             padding = torch.zeros(final_tor_len - window_data.shape[0], device=device)
#             window_data = torch.cat([window_data, padding])
        
#         # Truncate if necessary
#         window_data = window_data[:final_tor_len]

#         partitioned_data_single.append(window_data)
#         time_indices_single.append(time_end_idx)
#         size_indices_single.append(size_end_idx)
        
#     partitioned_data = torch.stack(partitioned_data_single, dim=0)
    
#     # Wrap in lists to maintain structure similar to AJSMAdeepcoffea
#     time_end_indices = [time_indices_single]
#     size_end_indices = [size_indices_single]

#     return partitioned_data, time_end_indices, size_end_indices
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
# 2. Core I-FGSM Attack Function
# ------------------------------------------------------------
def deepcoffea_ifgsm_attack(anchor_model, pandn_model, original_tor_window, original_exit_window,
                             time_end_idx, size_end_idx, device,
                             alpha_tensor, 
                             num_iter=100,
                             max_l2_ratio_time=0.2, max_l2_ratio_size=0.1):
    """
    一个保证只增加特征绝对值的定制化攻击。
    """
    anchor_model.eval()
    pandn_model.eval()

    adv_window = original_tor_window.clone().detach().unsqueeze(0).to(device)
    original_tor_window_dev = original_tor_window.clone().detach().unsqueeze(0).to(device)
    original_sign = torch.sign(original_tor_window_dev) # 预计算原始符号
    adv_window.requires_grad = True

    with torch.no_grad():
        exit_embedding = pandn_model(original_exit_window.clone().detach().unsqueeze(0).to(device))

        original_time_segment = original_tor_window_dev[:, 0 : time_end_idx + 1] if time_end_idx >= 0 else torch.tensor([], device=device)
        original_size_segment = original_tor_window_dev[:, time_end_idx + 1 : size_end_idx + 1] if size_end_idx > time_end_idx else torch.tensor([], device=device)
        
        original_time_norm = torch.linalg.norm(original_time_segment,ord = 2,dim = -1)
        original_size_norm = torch.linalg.norm(original_size_segment,ord = 2,dim = -1)
        
        absolute_max_l2_time = original_time_norm * max_l2_ratio_time
        absolute_max_l2_size = original_size_norm * max_l2_ratio_size

    for i in range(num_iter):
        adv_window.grad = None # 清除上一轮的梯度
        anchor_model.zero_grad()
            
        adv_embedding = anchor_model(adv_window)
        loss = F.cosine_similarity(adv_embedding.unsqueeze(0), exit_embedding.unsqueeze(0)).mean()
        loss.backward()

        if adv_window.grad is None:
            break
        
        grads = adv_window.grad.data
        grad_sign = grads.sign()

        # ### 为“增加绝对值”的更新逻辑 --- ###
        # 1. 计算理想的梯度下降步长
        step = -alpha_tensor * grad_sign
        # 2. 创建“同向”过滤器
        valid_update_mask = (torch.sign(step) == original_sign).float()
        # 3. 过滤掉会减小绝对值的步长
        filtered_step = step * valid_update_mask
        # 4. 保存上一步的更新量，以便在超标时回滚
        last_perturbation_step = filtered_step
        # 5. 应用最终扰动
        adv_window.data = adv_window.data + filtered_step
        # ### --- 修改结束 --- ###

        # --- 使用提前终止策略来执行约束 ---
        total_perturbation = adv_window.data - original_tor_window_dev
        
        current_time_pert = total_perturbation[:, 0 : time_end_idx + 1] if time_end_idx >= 0 else torch.tensor([], device=device)
        current_size_pert = total_perturbation[:, time_end_idx + 1 : size_end_idx + 1] if size_end_idx > time_end_idx else torch.tensor([], device=device)
        
        current_time_pert_norm = torch.linalg.norm(current_time_pert,ord = 2,dim = -1)
        current_size_pert_norm = torch.linalg.norm(current_size_pert,ord = 2,dim = -1)

        if (current_time_pert_norm/original_time_norm>max_l2_ratio_time) or \
           (current_size_pert_norm/original_size_norm>max_l2_ratio_size):
            # 回滚上一步操作
            adv_window.data = adv_window.data - last_perturbation_step 
            break
        
    total_perturbation = adv_window.data - original_tor_window_dev
    current_time_pert = total_perturbation[:, 0 : time_end_idx + 1] if time_end_idx >= 0 else torch.tensor([], device=device)
    current_size_pert = total_perturbation[:, time_end_idx + 1 : size_end_idx + 1] if size_end_idx > time_end_idx else torch.tensor([], device=device)
    
    current_time_pert_norm = torch.linalg.norm(current_time_pert.float(),ord=2,dim=-1)
    current_size_pert_norm = torch.linalg.norm(current_size_pert.float(),ord=2,dim=-1)
    time_ratio = (current_time_pert_norm/ original_time_norm).mean()
    size_ratio = (current_size_pert_norm / original_size_norm).mean()
    print(f"Stopping at iteration {i+1} due to L2 constraint.")
    print(f"Time perturbation ratio: {time_ratio.item():.4f}, Size perturbation ratio: {size_ratio.item():.4f}")  
    return adv_window.squeeze(0).detach()
# def deepcoffea_ifgsm_attack(anchor_model, pandn_model, original_tor_window, original_exit_window,
#                             time_end_idx, size_end_idx, device,
#                             epsilon_time=0.1, epsilon_size=0.01,
#                             num_iter=100,
#                             max_l2_ratio_time=0.2, max_l2_ratio_size=0.1):
#     """
#     Performs I-FGSM attack on a single DeepCoffea window pair to minimize cosine similarity.
#     """
#     anchor_model.eval()
#     pandn_model.eval()

#     adv_window = original_tor_window.clone().detach().unsqueeze(0).to(device)
#     original_tor_window_dev = original_tor_window.clone().detach().to(device)
#     adv_window.requires_grad = True

#     # Pre-calculate the embedding for the non-perturbed exit window
#     with torch.no_grad():
#         exit_embedding = pandn_model(original_exit_window.clone().detach().unsqueeze(0).to(device))

#     # Calculate L2 norm constraints
#     with torch.no_grad():
#         # Ensure indices are valid before slicing
#         original_time_segment = original_tor_window_dev[0 : time_end_idx + 1] if time_end_idx >= 0 else torch.tensor([], device=device)
#         original_size_segment = original_tor_window_dev[time_end_idx + 1 : size_end_idx + 1] if size_end_idx > time_end_idx else torch.tensor([], device=device)
        
#         original_time_norm = torch.linalg.norm(original_time_segment.float(),ord=2,dim=-1)
#         original_size_norm = torch.linalg.norm(original_size_segment.float(),ord=2,dim=-1)
        
#         epsilon = 1e-12
#         absolute_max_l2_time = original_time_norm * max_l2_ratio_time + epsilon
#         absolute_max_l2_size = original_size_norm * max_l2_ratio_size + epsilon

#     for i in range(num_iter):
#         anchor_model.zero_grad()
#         if adv_window.grad is not None:
#             adv_window.grad.zero_()
            
#         # Forward pass to get similarity
#         adv_embedding = anchor_model(adv_window)
#         # The "loss" we want to minimize is the cosine similarity
#         loss = F.cosine_similarity(adv_embedding.unsqueeze(0), exit_embedding.unsqueeze(0))

#         # Backward pass to get gradients
#         loss.backward()
#         grads = adv_window.grad.data

#         # Create step-size tensor based on feature type
#         step_size = torch.zeros_like(grads)
#         if time_end_idx >= 0:
#             step_size[:, 0 : time_end_idx + 1] = epsilon_time
#         if size_end_idx > time_end_idx:
#             step_size[:, time_end_idx + 1 : size_end_idx + 1] = epsilon_size

#         # I-FGSM update rule to MINIMIZE the loss (similarity)
#         # x_adv = x - epsilon * sign(gradient)
#         perturbation = step_size * grads.sign()
#         adv_window.data = adv_window.data - perturbation

#         # --- Enforce L2 norm constraint ---
#         total_perturbation = adv_window.data - original_tor_window_dev
        
#         current_time_pert = total_perturbation[:, 0 : time_end_idx + 1] if time_end_idx >= 0 else torch.tensor([], device=device)
#         current_size_pert = total_perturbation[:, time_end_idx + 1 : size_end_idx + 1] if size_end_idx > time_end_idx else torch.tensor([], device=device)
        
#         current_time_pert_norm = torch.linalg.norm(current_time_pert.float(),ord=2,dim=-1)
#         current_size_pert_norm = torch.linalg.norm(current_size_pert.float(),ord=2,dim=-1)

#         # If budget is exceeded, revert the last step and break
#         if (current_time_pert_norm/original_time_norm>max_l2_ratio_time) or \
#            (current_size_pert_norm/original_size_norm>max_l2_ratio_size):
#             adv_window.data = adv_window.data + perturbation # revert
#             break
#     total_perturbation = adv_window.data - original_tor_window_dev
#     current_time_pert = total_perturbation[:, 0 : time_end_idx + 1] if time_end_idx >= 0 else torch.tensor([], device=device)
#     current_size_pert = total_perturbation[:, time_end_idx + 1 : size_end_idx + 1] if size_end_idx > time_end_idx else torch.tensor([], device=device)
    
#     current_time_pert_norm = torch.linalg.norm(current_time_pert.float(),ord=2,dim=-1)
#     current_size_pert_norm = torch.linalg.norm(current_size_pert.float(),ord=2,dim=-1)
#     time_ratio = (current_time_pert_norm/ original_time_norm).mean()
#     size_ratio = (current_size_pert_norm / original_size_norm).mean()
#     print(f"Stopping at iteration {i+1} due to L2 constraint.")
#     print(f"Time perturbation ratio: {time_ratio.item():.4f}, Size perturbation ratio: {size_ratio.item():.4f}")    
#     return adv_window.squeeze(0).detach()


# ------------------------------------------------------------
# 3. Main Execution Function
# ------------------------------------------------------------
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load DeepCoffea Model ---
    print("Loading pre-trained DeepCoffea model...")
    ckpt_path = pathlib.Path(args.target_model_path).resolve()
    # Extract parameters from model path name
    fields = ckpt_path.name.split("_")
    emb_size = int(fields[-4].split("es")[-1])
    tor_len = int(fields[-8].split("tl")[-1])
    exit_len = int(fields[-7].split("el")[-1])

    anchor_model = DeepCoffeaModel(emb_size=emb_size, input_size=tor_len * 2).to(device)
    pandn_model = DeepCoffeaModel(emb_size=emb_size, input_size=exit_len * 2).to(device)

    model_checkpoint_path = os.path.join(args.target_model_path, 'best_loss.pth')
    state_dict = torch.load(model_checkpoint_path, map_location=device)
    anchor_model.load_state_dict(state_dict['anchor_state_dict'])
    pandn_model.load_state_dict(state_dict['pandn_state_dict'])
    anchor_model.eval()
    pandn_model.eval()
    print("Model loaded successfully.")

    # --- Load Dataset ---
    print("Loading dataset...")
    data_filename = f"d{args.delta}_ws{args.win_size}_nw{args.n_wins}_thr{args.threshold}_tl{args.tor_len}_el{args.exit_len}_nt{args.n_test}"
    
    session_path = os.path.join(args.data_path, "filtered_and_partitioned", f"{data_filename}_test_session.npz")
    win_path = os.path.join(args.data_path, "filtered_and_partitioned", f"{data_filename}_test.npz")

    test_data_session = np.load(session_path, allow_pickle=True)
    test_data_win = np.load(win_path)
    
    indices_to_attack = np.arange(args.num_samples)

    # --- Storage for results ---
    original_sims = []
    attacked_sims = []
    time_l2_ratios = []
    size_l2_ratios = []
    total_adv_tor_windows = []
    total_original_tor_windows = []
    total_original_exit_windows = []
    all_original_sims, all_attacked_sims = [], []
    total_repartitioned_adv_windows = []
    print(f"Starting I-FGSM attack on the first {args.num_samples} samples...")
    for idx in tqdm(indices_to_attack):
        tor_ipd = torch.tensor(test_data_session['tor_ipds'][idx], dtype=torch.float32).to(device)
        tor_size = torch.tensor(test_data_session['tor_sizes'][idx], dtype=torch.float32).to(device)
        original_session = torch.stack([tor_ipd, tor_size], dim=0)
        original_exit_windows = torch.tensor(test_data_win['test_exit'][:,idx,:], dtype=torch.float32).to(device)

        original_tor_windows, time_indices, size_indices,boundaries = partition_single_session(
            original_session, args.delta, args.win_size, args.n_wins, args.tor_len, device)
        
        # time_end_indices = time_indices_list[0]
        # size_end_indices = size_indices_list[0]
        
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

            adv_win = deepcoffea_ifgsm_attack(
                anchor_model, pandn_model, tor_win, exit_win,
                t_end_idx, s_end_idx, device,
                alpha_tensor, args.num_iter,
                args.max_ratio_time, args.max_ratio_size,
                
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

            adv_tor_emb = anchor_model(repartitioned_adv_windows)
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
    print("\n--- Custom IFGSM Attack on DeepCoffea: Final Results ---")
    print(f"Attacked Samples: {args.num_samples}")
    print(f"Average Similarity BEFORE Attack: {np.mean(all_original_sims):.4f}")
    print(f"Average Similarity AFTER Attack:  {np.mean(all_attacked_sims):.4f}")
    print("-" * 20)
    print(f" Average L2 Perturbation Ratio (Time): {np.mean(time_l2_ratios):.4f}")
    print(f" Average L2 Perturbation Ratio (Size): {np.mean(size_l2_ratios):.4f}")
    result_fpath = pathlib.Path(f'baseline/I_FGSM/IFGSMdeepcoffea_advsamples_time{np.mean(time_l2_ratios):.4f}_size{np.mean(size_l2_ratios):.4f}.p')
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
    parser = argparse.ArgumentParser(description="Custom IFGSM attack on DeepCoffea")
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
    parser.add_argument("--num_iter", type=int, default=100, help="IFGSM iterations.")
    parser.add_argument("--max_ratio_time", type=float, default=0.15, help="Max L2 ratio for time features.")
    parser.add_argument("--max_ratio_size", type=float, default=0.15, help="Max L2 ratio for size features.")
    parser.add_argument("--alpha_time", type=float, default=1, help="Step size for time features.")
    parser.add_argument("--alpha_size", type=float, default=0.02, help="Step size for size features.")
    
    args = parser.parse_args()
    main(args)