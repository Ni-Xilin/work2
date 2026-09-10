import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import pathlib
import pickle

# 从 deepcoffea.py 引入模型定义
from target_model.Deepcoffea import Model as DeepCoffeaModel

# 从 xxx.py 和 RMdeepcoffea.py 引入的数据处理逻辑和我们创建的函数
from data_provider.data_loader import DeepCoffeaDataset
from torch.utils.data import DataLoader

# ------------------------------------------------------------
# 1. 我们之前创建的、用于切分、还原窗口并获取索引的函数
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
# def partition_sessions_with_end_indices(sessions, delta, win_size, n_wins, tor_len, device):
#     """
#     按照 IPD 划分 adv_sessions 的窗口，将每个窗口的 IPD 第一个元素改为 0。
#     同时，此函数会计算时间和大小数据的L2范数，并返回它们在最终窗口张量中的精确索引。

#     参数:
#         adv_sessions (list of torch.Tensor): 每段会话的扰动后数据，每个会话的形状为 (2, session_length)。
#         delta (int): 窗口重叠时间（秒）。
#         win_size (int): 每个窗口的大小（秒）。
#         n_wins (int): 每段会话的窗口数量。
#         tor_len (int): 每个窗口中 IPD 和 size 各自的目标长度。最终窗口长度为 tor_len * 2。
#         device: 计算设备 (例如, 'cuda:0' 或 'cpu')。

#     返回:
#         partitioned_data (torch.Tensor): 划分并处理后的窗口数据，形状为 [batch_size, n_wins, tor_len * 2]。
#         time_l2_norms (list): 包含每个窗口时间数据L2范数的列表。
#         size_l2_norms (list): 包含每个窗口大小数据L2范数的列表。
#         time_indices (list): 包含每个窗口时间数据索引的列表。
#         size_indices (list): 包含每个窗口大小数据索引的列表。
#     """
#     win_size_ms = win_size * 1000
#     delta_ms = delta * 1000
#     offset = win_size_ms - delta_ms

#     partitioned_data_list = []
#     time_end_indices = []
#     size_end_indices = []

#     # sessions 现在是单个会话
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
        
#         time_end_idx = len_ipd - 1 if len_ipd > 0 else 0
#         size_end_idx = (len_ipd + len_size - 1) if len_size > 0 else 0
        
#         final_tor_len = tor_len * 2
#         time_end_idx = min(time_end_idx, final_tor_len - 1)
#         size_end_idx = min(size_end_idx, final_tor_len - 1)

#         window = torch.cat([window_ipd, window_size])

#         if window.shape[0] < final_tor_len:
#             padding = torch.zeros(final_tor_len - window.shape[0], device=device)
#             window = torch.cat([window, padding])
#         window = window[:final_tor_len]

#         partitioned_data_single.append(window)
#         time_indices_single.append(time_end_idx)
#         size_indices_single.append(size_end_idx)
        
#     partitioned_data = torch.stack(partitioned_data_single, dim=0)
#     time_end_indices.append(time_indices_single)
#     size_end_indices.append(size_indices_single)

#     return partitioned_data, time_end_indices, size_end_indices

# ------------------------------------------------------------
# 2. 核心攻击函数
# ------------------------------------------------------------


def deepcoffea_jsma_attack(anchor_model, pandn_model, original_tor_window, original_exit_window,
                           time_end_idx, size_end_idx, device,
                           epsilon_time=0.1, epsilon_size=0.01,
                           max_features_to_perturb=100,
                           max_l2_ratio_time=0.2, max_l2_ratio_size=0.1):
    """
    对单个流量窗口执行最终版的JSMA风格攻击。
    选择逻辑: 寻找能让分数下降最快的点。
    扰动逻辑: 始终只增大特征的绝对值。
    """
    anchor_model.eval()
    pandn_model.eval()

    adv_window = original_tor_window.clone().detach().unsqueeze(0).to(device)
    adv_window.requires_grad = True
    
    with torch.no_grad():
        exit_embedding = pandn_model(original_exit_window.clone().detach().unsqueeze(0).to(device))

    with torch.no_grad():
        original_time_segment = original_tor_window[0 : time_end_idx + 1]
        original_size_segment = original_tor_window[time_end_idx + 1 : size_end_idx + 1]
        original_time_norm = torch.linalg.norm(original_time_segment.float(),ord=2,dim = -1).mean()
        original_size_norm = torch.linalg.norm(original_size_segment.float(),ord=2,dim = -1).mean()
        epsilon = 1e-12
        absolute_max_l2_time = original_time_norm * max_l2_ratio_time + epsilon
        absolute_max_l2_size = original_size_norm * max_l2_ratio_size + epsilon

    perturbed_features_mask = torch.zeros_like(original_tor_window).bool()

    for i in range(max_features_to_perturb):
        adv_embedding = anchor_model(adv_window)
        current_sim = F.cosine_similarity(adv_embedding.unsqueeze(0), exit_embedding.unsqueeze(0))
        
        anchor_model.zero_grad()
        if adv_window.grad is not None:
            adv_window.grad.zero_()
        current_sim.backward()
        grads = adv_window.grad.data.squeeze().detach()
        
        # ##############################################################

        # 1. 扰动方式：只增大特征绝对值。这意味着扰动方向和原始值符号相同。
        #    扰动量 perturbation = epsilon_step * sign(original_value)
        # 2. 选择标准：找到能让分数下降最多的点。
        #    分数变化量 ≈ grad * perturbation = grad * (epsilon_step * sign(original_value))
        # 我们要让这个变化量最“负”，等价于让 grad * sign(original_value) 最“负”。
        # 所以，显著图 saliency = -(grad * sign(original_value))，我们找这个图里的最大值。
        
        saliency_map = -grads * torch.sign(original_tor_window)
        
        # 过滤掉不好的攻击点（saliency < 0 的点）
        saliency_map[saliency_map < 0] = -float('inf')
        # ##############################################################

        # 应用其他屏蔽
        saliency_map[perturbed_features_mask] = -float('inf')
        saliency_map[size_end_idx + 1:] = -float('inf')

        # 全局寻找最优的点
        best_saliency_val, best_feature_idx = saliency_map.max(0)
        
        if torch.isinf(best_saliency_val) or best_saliency_val == 0:
            break
            
        perturbed_features_mask[best_feature_idx] = True
        
        if best_feature_idx <= time_end_idx:
            epsilon_step = epsilon_time
        else:
            epsilon_step = epsilon_size

        # ##############################################################
        # ### 确保只增大绝对值 ###
        # ##############################################################
        original_feature_value = original_tor_window[best_feature_idx].item()
        
        # 1. 获取原始值的符号
        feature_sign = torch.sign(torch.tensor(original_feature_value))
        if feature_sign == 0:
            continue  # 如果原始值为0，跳过这个特征
            
        # 2. 计算带有正确方向的扰动
        perturbation_with_direction = feature_sign * epsilon_step
        # ##############################################################

        potential_adv_window = adv_window.clone().detach()
        potential_adv_window[0, best_feature_idx] += perturbation_with_direction
        
        total_perturbation = potential_adv_window.squeeze(0) - original_tor_window
        current_time_pert = total_perturbation[0 : time_end_idx + 1]
        current_size_pert = total_perturbation[time_end_idx + 1 : size_end_idx + 1]
        
        if torch.linalg.norm(current_time_pert.float(),ord =2,dim = -1).mean() > absolute_max_l2_time or \
           torch.linalg.norm(current_size_pert.float(),ord =2,dim = -1).mean() > absolute_max_l2_size:
            perturbed_features_mask[best_feature_idx] = False
            break

        adv_window.data = potential_adv_window.data
    print("攻击完成，扰动特征数量:", perturbed_features_mask.sum().item())
    final_perturbation = adv_window.squeeze(0).detach() - original_tor_window
    return adv_window.squeeze(0).detach(), final_perturbation


# ------------------------------------------------------------
# 3. 主函数
# ------------------------------------------------------------
def main(args):
    # --- 设备设置 ---
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 加载预训练的 DeepCoffea 模型 ---
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
    test_session_path = os.path.join(args.data_path, "filtered_and_partitioned", 
                                     f"d{args.delta}_ws{args.win_size}_nw{args.n_wins}_thr{args.threshold}_tl{args.tor_len}_el{args.exit_len}_nt{args.n_test}_test_session.npz")
    test_data_session = np.load(test_session_path, allow_pickle=True)

    # 攻击前N个样本
    num_samples_to_attack = 500
    indices_to_attack = np.arange(num_samples_to_attack)

    original_sims = []
    attacked_sims = []
    time_l2_ratios = []
    size_l2_ratios = []
    total_adv_tor_windows = []
    total_original_tor_windows = []
    total_original_exit_windows = []

    print(f"开始攻击前 {num_samples_to_attack} 个样本...")
    for idx in tqdm(indices_to_attack):
        # --- 准备单个会话的数据 ---
        tor_ipd = torch.tensor(test_data_session['tor_ipds'][idx]).to(device)
        tor_size = torch.tensor(test_data_session['tor_sizes'][idx]).to(device)
        original_session = torch.stack([tor_ipd, tor_size], dim=0)

        test_win_path = test_session_path.replace('_session.npz', '.npz')
        test_data_win = np.load(test_win_path)
        original_exit_windows = torch.tensor(test_data_win['test_exit'][:,idx,:]).float().to(device)

        # --- 对原始会话进行切分，得到窗口和索引 ---
        original_tor_windows, time_indices, size_indices,boundaries = partition_single_session(
            original_session, args.delta, args.win_size, args.n_wins, args.tor_len, device)
        
        # time_end_indices = time_end_indices_list[0]
        # size_end_indices = size_end_indices_list[0]
        
        adv_tor_windows = []
        
        # --- 逐个窗口进行攻击 ---
        for i in range(args.n_wins):
            tor_win = original_tor_windows[i].float()
            exit_win = original_exit_windows[i].float()
            t_end_idx = time_indices[i]
            s_end_idx = size_indices[i]
            
     
            adv_win, _ = deepcoffea_jsma_attack(
                anchor_model, pandn_model, tor_win, exit_win, t_end_idx, s_end_idx, device,
                epsilon_time=args.epsilon_time,
                epsilon_size=args.epsilon_size,
                max_features_to_perturb=args.max_features_to_perturb,
                max_l2_ratio_time=args.max_l2_ratio_time,
                max_l2_ratio_size=args.max_l2_ratio_size
            )
            adv_tor_windows.append(adv_win)
        adv_tor_windows = torch.stack(adv_tor_windows, dim=0).float()
        reverted_session = reconstruct_single_session(
                    adv_tor_windows, original_session, 
                    boundaries, time_indices, size_indices
            )
        repartitioned_adv_windows, _, _, _ = partition_single_session(
                reverted_session, args.delta, args.win_size, args.n_wins, args.tor_len, device
            )
        # --- 评估攻击效果 ---
        with torch.no_grad():

            
            orig_tor_emb = anchor_model(original_tor_windows.float())
            orig_exit_emb = pandn_model(original_exit_windows.float())
            orig_sim = F.cosine_similarity(orig_tor_emb, orig_exit_emb).mean().item()
            original_sims.append(orig_sim)

            adv_tor_emb = anchor_model(adv_tor_windows)
            attacked_sim = F.cosine_similarity(adv_tor_emb, orig_exit_emb).mean().item()
            attacked_sims.append(attacked_sim)

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
            print(f"时间特征L2扰动比率: {np.mean(session_time_ratios):.4f}, 大小特征L2扰动比率: {np.mean(session_size_ratios):.4f}")
            time_l2_ratios.append(np.mean(session_time_ratios))
            size_l2_ratios.append(np.mean(session_size_ratios))
            total_adv_tor_windows.append(repartitioned_adv_windows.cpu())
            total_original_tor_windows.append(original_tor_windows.cpu())
            total_original_exit_windows.append(original_exit_windows.cpu())

    # --- 保存结果 ---
    # 注意：现在保存的list长度是 num_samples_to_attack
    time_l2_ratios = np.mean(time_l2_ratios).item()
    size_l2_ratios = np.mean(size_l2_ratios).item()
    original_sims = np.mean(original_sims).item()
    attacked_sims = np.mean(attacked_sims).item()
    result_fpath = pathlib.Path(f'baseline/AJSMA/AJSMAdeepcoffea_advsamples_time{time_l2_ratios:.4f}_size{size_l2_ratios:.4f}.p')
    with open(result_fpath, "wb") as fp:
        results = {
            "adv_tor_windows": total_adv_tor_windows,
            "original_tor_windows": total_original_tor_windows,
            "original_exit_windows": total_original_exit_windows,
            "time_l2_ratios": time_l2_ratios,
            "size_l2_ratios": size_l2_ratios,
            "original_sims": original_sims,
            "attacked_sims": attacked_sims
        }
        pickle.dump(results, fp)
    print(f"\n攻击样本和结果已保存到: {result_fpath}")

    # --- 打印最终结果 ---
    print("\n--- 攻击评估结果 ---")
    print(f"攻击样本总数: {num_samples_to_attack}")
    print(f"平均原始相似度: {original_sims:.4f}")
    print(f"平均攻击后相似度: {attacked_sims:.4f}")
    print("-" * 20)
    print(f"平均时间特征L2扰动比率: {time_l2_ratios:.4f}")
    print(f"平均大小特征L2扰动比率: {size_l2_ratios:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AJSMA-style attack on DeepCoffea")
    # --- 数据和模型参数 ---
    parser.add_argument("--target_model_path", default="target_model/deepcoffea/deepcoffea_d3_ws5_nw11_thr20_tl500_el800_nt1000_ap1e-01_es64_lr1e-03_mep100000_bs256/",type=str, help="指向预训练DeepCoffea模型文件夹的路径")
    parser.add_argument("--data_path",  default="target_model/deepcoffea/dataset/CrawlE_Proc/",type=str,help="指向数据集根目录的路径")
    parser.add_argument("--device",  default="cuda:0", type=str,help="计算设备, e.g., 'cuda:0' or 'cpu'")
    # 如下参数需要和数据集文件名匹配
    parser.add_argument("--delta", default=3, type=int)
    parser.add_argument("--win_size", default=5, type=int)
    parser.add_argument("--n_wins", default=11, type=int)
    parser.add_argument("--threshold", default=20, type=int)
    parser.add_argument("--tor_len", default=500, type=int)
    parser.add_argument("--exit_len", default=800, type=int)
    parser.add_argument("--n_test", default=1000, type=int)

    # --- 攻击超参数 ---
    parser.add_argument("--epsilon_time", type=float, default=5, help="对时间特征的单步固定扰动值")
    parser.add_argument("--epsilon_size", type=float, default=0.1, help="对大小特征的单步固定扰动值")
    # parser.add_argument("--epsilon_time", type=float, default=25, help="对时间特征的单步固定扰动值")
    # parser.add_argument("--epsilon_size", type=float, default=1, help="对大小特征的单步固定扰动值")
    parser.add_argument("--max_features_to_perturb", type=int, default=300, help="每个窗口最多修改的特征点数")
    parser.add_argument("--max_l2_ratio_time", type=float, default=0.15, help="时间特征总扰动的最大L2范数比率")
    parser.add_argument("--max_l2_ratio_size", type=float, default=0.15, help="大小特征总扰动的最大L2范数比率")

    args = parser.parse_args()
    main(args)