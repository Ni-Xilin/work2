import argparse
import os
import pathlib
import pickle
import random
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm

# 导入 DeepCoffea 模型定义
from target_model.Deepcoffea import Model as DeepCoffeaModel

# --- 为了可复现性，设置随机种子 ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ------------------------------------------------------------
# 1. 从训练脚本中移植必要的模块和函数定义
#    在实际使用中，您可以将这些定义放在一个共享的 utils.py 文件中并导入
# ------------------------------------------------------------

# --- 生成器模型定义 ---
class TIMENOISER(nn.Module):
    """生成时序扰动的模型。"""
    def __init__(self, in_size):
        super(TIMENOISER, self).__init__()
        self.network = nn.Sequential(nn.Linear(in_size, 512), nn.ReLU(), nn.Linear(512, in_size))

    def forward(self, z, time_end_idx_batch, sigma, mid):
        raw_pert = self.network(z)
        # 对批次中的每个样本独立地进行标准化
        res = raw_pert - raw_pert.mean(dim=-1, keepdim=True) + mid
        res_std = res.std(dim=-1, keepdim=True)
        res = res * (sigma / (res_std + 1e-9))
        
        final_pert = torch.zeros_like(raw_pert)
        # 为批次中的每个窗口应用其专属的掩码
        for i in range(z.shape[0]):
            time_end_idx = time_end_idx_batch[i]
            if time_end_idx >= 0:
                safe_slice_end = min(time_end_idx + 1, final_pert.shape[1])
                final_pert[i, :safe_slice_end] = res[i, :safe_slice_end]
        return final_pert

class ADDNOISER(nn.Module):
    """生成用于数据包注入的位置和大小噪声的模型。"""
    def __init__(self, in_size):
        super(ADDNOISER, self).__init__()
        self.where_net = nn.Sequential(nn.Linear(in_size, 512), nn.ReLU(), nn.Linear(512, in_size))
        self.size_net = nn.Sequential(nn.Linear(in_size, 512), nn.ReLU(), nn.Linear(512, in_size))
    def forward(self, z):
        return self.where_net(z), self.size_net(z)

class SIZEPADNOISER(nn.Module):
    """生成用于大小填充的位置噪声的模型。"""
    def __init__(self, in_size):
        super(SIZEPADNOISER, self).__init__()
        self.network = nn.Sequential(nn.Linear(in_size, 512), nn.ReLU(), nn.Linear(512, in_size))
    def forward(self, z):
        return self.network(z)

# --- 数据处理函数 ---
def partition_single_session(session, delta, win_size, n_wins, tor_len, device):
    """
    (来自PGDdeepcoffea.py) 划分单条会话，生成窗口、特征结束索引和原始边界。
    """
    win_size_ms = win_size * 1000
    delta_ms = delta * 1000
    offset = win_size_ms - delta_ms
    session_ipd = session[0, :]
    session_size = session[1, :]
    cumulative_time = session_ipd.abs().cumsum(dim=0)
    
    partitioned_data_single, time_indices_single, size_indices_single, original_boundaries_single = [], [], [], []

    for wi in range(int(n_wins)):
        start_time, end_time = wi * offset, wi * offset + win_size_ms
        start_idx, end_idx = torch.searchsorted(cumulative_time, start_time).item(), torch.searchsorted(cumulative_time, end_time).item()
        original_boundaries_single.append((start_idx, end_idx))
        
        window_ipd, window_size = session_ipd[start_idx:end_idx], session_size[start_idx:end_idx]

        if len(window_ipd) > 0:
            window_ipd = torch.cat([torch.tensor([0.0]).to(device), window_ipd[1:]])

        len_ipd, len_size = len(window_ipd), len(window_size)
        time_end_idx, size_end_idx = (len_ipd - 1 if len_ipd > 0 else -1), ((len_ipd + len_size - 1) if len_size > 0 else -1)
        
        final_tor_len = tor_len * 2
        window_data = torch.cat([window_ipd, window_size])
        if window_data.shape[0] < final_tor_len:
            padding = torch.zeros(final_tor_len - window_data.shape[0], device=device)
            window_data = torch.cat([window_data, padding])
        window_data = window_data[:final_tor_len]

        partitioned_data_single.append(window_data)
        time_indices_single.append(time_end_idx)
        size_indices_single.append(size_end_idx)
    
    return torch.stack(partitioned_data_single, dim=0), time_indices_single, size_indices_single, original_boundaries_single

def reconstruct_single_session(adv_windows_tensor, original_session, boundaries, time_indices, size_indices):
    """【最终修正版】根据扰动后的窗口和原始边界，通过动态扩展会话长度来还原完整的对抗会话流。"""
    reconstructed_session = original_session.clone()
    for j in range(adv_windows_tensor.shape[0]):
        adv_win, (s_idx, e_idx), t_end, s_end = adv_windows_tensor[j], boundaries[j], time_indices[j], size_indices[j]
        if s_idx >= e_idx and t_end < 0: continue
        
        len_ipd, len_size = t_end + 1, s_end - t_end
        adv_ipd, adv_size = adv_win[:len_ipd], adv_win[len_ipd : len_ipd + len_size]
        
        # 动态扩展会话张量以容纳更长的窗口
        if len(adv_ipd) > 1:
            needed_len = s_idx + len(adv_ipd)
            if needed_len > reconstructed_session.shape[1]:
                padding = torch.zeros((2, needed_len - reconstructed_session.shape[1]), device=reconstructed_session.device)
                reconstructed_session = torch.cat([reconstructed_session, padding], dim=1)
            reconstructed_session[0, s_idx + 1 : needed_len] = adv_ipd[1:]

        if len(adv_size) > 0:
            needed_len = s_idx + len(adv_size)
            if needed_len > reconstructed_session.shape[1]:
                padding = torch.zeros((2, needed_len - reconstructed_session.shape[1]), device=reconstructed_session.device)
                reconstructed_session = torch.cat([reconstructed_session, padding], dim=1)
            reconstructed_session[1, s_idx : needed_len] = adv_size
            
    return reconstructed_session

# --- 自定义Autograd函数 (在评估时不需要梯度，但为保持一致性而保留) ---
class PacketInjectionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, noise_where, noise_size, data_in, num_to_add, time_end_idx_batch, size_end_idx_batch):
        # 评估模式下，我们直接在 apply_blanket_attack_with_alignment 中实现逻辑
        # 此处仅为保持与训练代码的API一致
        return data_in
class SizePaddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw_perturbation, data_in, time_end_idx_batch, size_end_idx_batch, total_overhead_bytes, per_packet_overhead_bytes):
        batch_size = data_in.shape[0]
        data_out = data_in.clone()
        if total_overhead_bytes == 0: return data_out
        for i in range(batch_size):
            time_end_idx = time_end_idx_batch[i]
            max_len = data_out.shape[1]
            if time_end_idx + 1 >= max_len: continue
            valid_size_len = max_len - (time_end_idx + 1)
            if valid_size_len <= 0: continue
            remaining_overhead = total_overhead_bytes
            indices_to_perturb = torch.argsort(raw_perturbation[i, :valid_size_len], descending=True)
            for j in indices_to_perturb:
                if remaining_overhead <= 0: break
                delta = min(per_packet_overhead_bytes, remaining_overhead)
                remaining_overhead -= delta
                actual_idx = time_end_idx + 1 + j
                original_sign = torch.sign(data_out[i, actual_idx])
                if original_sign == 0: original_sign = 1
                data_out[i, actual_idx] += (original_sign * (delta / 1000.0))
        return data_out
    @staticmethod
    def backward(ctx, grad_output): return None, None, None, None, None, None

# --- 辅助函数与别名 ---
def generate_perturbation(change_points, size):
    """
    (来自BLANKETdeepcorrtest.py) 生成一个排列图谱 (permutation map)，用于移动数据包为注入腾出空间。
    """
    if isinstance(change_points, torch.Tensor):
        change_points = change_points.cpu().numpy()
    start, pert, passed = size - len(change_points), [], 0
    change_points_set = set(change_points.tolist())
    for ind in range(size):
        if ind in change_points_set:
            pert.append(start); start += 1; passed += 1
        else:
            pert.append(ind - passed)
    return pert

pad_sizes = SizePaddingFunction.apply 

# ------------------------------------------------------------
# 2. 核心攻击与L2计算函数 (插入式)
# ------------------------------------------------------------
@torch.no_grad()
def apply_blanket_attack_with_alignment(original_win, metadata, generators, args, device):
    """
    对单个窗口应用“插入式”BLANKET攻击，并返回对齐后的原始样本以便精确计算L2范数。
    """
    time_noiser, add_noiser, pad_noiser = generators
    time_end_idx, size_end_idx = metadata
    
    original_win_batch = original_win.unsqueeze(0)
    z = torch.randn(1, args.tor_len * 2, device=device)
    where_batch, sizes_batch = add_noiser(z)
    pad_noise_batch = pad_noiser(z)
    time_pert_batch = time_noiser(z, [time_end_idx], args.sigma, args.mid)
    where, sizes, pad_noise, time_pert = where_batch[0], sizes_batch[0], pad_noise_batch[0], time_pert_batch

    aligned_original = original_win_batch.clone()
    adv_win_temp = original_win_batch.clone()
    
    # --- 核心“分块对齐”与插入 ---
    safe_ipd_len = min(time_end_idx + 1, original_win.shape[0])
    tops = torch.tensor([], dtype=torch.long, device=device)
    if safe_ipd_len > 0 and args.to_add > 0:
        num_to_inject = int(min(args.to_add, safe_ipd_len))
        tops = torch.argsort(where[:safe_ipd_len], descending=False)[:num_to_inject]
        
        # a. 只对IPD块进行排列
        ipd_perts_map = generate_perturbation(tops, size=safe_ipd_len)
        ipd_perts_tensor = torch.tensor(ipd_perts_map, device=device)
        aligned_original[0, :safe_ipd_len] = aligned_original[0, :safe_ipd_len][ipd_perts_tensor]
        adv_win_temp[0, :safe_ipd_len] = adv_win_temp[0, :safe_ipd_len][ipd_perts_tensor]
        
        # b. 在对齐后的原始样本中，将注入位清零
        aligned_original[0, tops] = 0.0

        # c. 注入新的IPD和Size特征
        original_valid_data_len = time_end_idx + 1
        size_indices = original_valid_data_len + tops
        mask = size_indices < original_win.shape[0]
        final_time_indices, final_size_indices = tops[mask], size_indices[mask]
        if len(final_time_indices) > 0:
            time_signs = torch.sign(adv_win_temp[0, final_time_indices]); time_signs[time_signs==0] = 1
            size_signs = torch.sign(adv_win_temp[0, final_size_indices]); size_signs[size_signs==0] = 1
            original_size_val = 0.595 * (torch.sigmoid(sizes[tops]))
            final_size_val = original_size_val[mask]
            final_time_val = torch.full_like(final_size_val, 1.0)
            adv_win_temp[0, final_time_indices] = time_signs * final_time_val
            adv_win_temp[0, final_size_indices] = size_signs * final_size_val
    
    # --- 后续扰动步骤 ---
    adv_win_padded = pad_sizes(pad_noise_batch, adv_win_temp, [time_end_idx], [size_end_idx], 
                               args.size_padding_total_kb * 1024, args.size_padding_packet_bytes)
    
    original_sign = torch.sign(adv_win_padded)
    signed_pert = original_sign * torch.abs(time_pert_batch)
    final_adv_win = adv_win_padded + signed_pert

    return final_adv_win.squeeze(0), aligned_original.squeeze(0), len(tops)

# ------------------------------------------------------------
# 3. 主评估函数
# ------------------------------------------------------------
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 模型加载 ---
    print("正在加载预训练模型...")
    anchor_model = DeepCoffeaModel(emb_size=args.emb_size, input_size=args.tor_len * 2).to(device)
    pandn_model = DeepCoffeaModel(emb_size=args.emb_size, input_size=args.exit_len * 2).to(device)
    model_path = os.path.join(args.target_model_path, 'best_loss.pth')
    state_dict = torch.load(model_path, map_location=device)
    anchor_model.load_state_dict(state_dict['anchor_state_dict'])
    pandn_model.load_state_dict(state_dict['pandn_state_dict'])
    anchor_model.eval(); pandn_model.eval()

    time_noiser = TIMENOISER(args.tor_len * 2).to(device)
    add_noiser = ADDNOISER(args.tor_len * 2).to(device)
    pad_noiser = SIZEPADNOISER(args.tor_len * 2).to(device)
    blanket_checkpoint = torch.load(args.blanket_model_path, map_location=device)
    time_noiser.load_state_dict(blanket_checkpoint['time_noiser_state_dict'])
    add_noiser.load_state_dict(blanket_checkpoint['add_noiser_state_dict'])
    pad_noiser.load_state_dict(blanket_checkpoint['pad_noiser_state_dict'])
    time_noiser.eval(); add_noiser.eval(); pad_noiser.eval()
    print("所有模型加载成功。")

    # --- 数据集加载 ---
    print("正在加载测试数据集...")
    data_filename = f"d{args.delta}_ws{args.win_size}_nw{args.n_wins}_thr{args.threshold}_tl{args.tor_len}_el{args.exit_len}_nt{args.n_test}"
    session_path = os.path.join(args.data_path, "filtered_and_partitioned", f"{data_filename}_test_session.npz")
    win_path = os.path.join(args.data_path, "filtered_and_partitioned", f"{data_filename}_test.npz")
    test_sessions = np.load(session_path, allow_pickle=True)
    test_wins = np.load(win_path, allow_pickle=True)
    
    all_original_sims, all_attacked_sims, all_time_ratios, all_size_ratios = [], [], [], []
    total_repartitioned_adv_windows, total_original_tor_windows, total_original_exit_windows = [], [], []

    print(f"开始对 {args.num_samples} 个样本进行BLANKET攻击...")
    for idx in tqdm(range(args.num_samples), desc="攻击样本中"):
        tor_ipd = torch.tensor(test_sessions['tor_ipds'][idx], dtype=torch.float32, device=device)
        tor_size = torch.tensor(test_sessions['tor_sizes'][idx], dtype=torch.float32, device=device)
        original_session = torch.stack([tor_ipd, tor_size], dim=0)
        original_exit_windows = torch.tensor(test_wins['test_exit'][:, idx, :], dtype=torch.float32, device=device)
        
        original_tor_windows, time_indices, size_indices, boundaries = partition_single_session(
            original_session, args.delta, args.win_size, args.n_wins, args.tor_len, device)
        
        adv_wins_list, aligned_orig_wins_list, num_injected_list = [], [], []
        for i in range(args.n_wins):
            if time_indices[i] < 0:
                adv_wins_list.append(original_tor_windows[i])
                aligned_orig_wins_list.append(original_tor_windows[i])
                num_injected_list.append(0)
                continue
            metadata = (time_indices[i], size_indices[i])
            adv_win, aligned_orig, num_injected = apply_blanket_attack_with_alignment(
                original_tor_windows[i], metadata, (time_noiser, add_noiser, pad_noiser), args, device)
            adv_wins_list.append(adv_win)
            aligned_orig_wins_list.append(aligned_orig)
            num_injected_list.append(num_injected)

        adv_tor_windows = torch.stack(adv_wins_list)
        aligned_tor_windows = torch.stack(aligned_orig_wins_list)

        # --- 【核心修正】使用动态边界计算L2比率 ---
        pure_perturbation = adv_tor_windows - aligned_tor_windows
        session_time_ratios, session_size_ratios = [], []
        for i in range(args.n_wins):
            t_end_idx_orig, s_end_idx_orig = time_indices[i], size_indices[i]
            if t_end_idx_orig < 0: continue
            
            # 计算注入后的新边界
            num_inj = num_injected_list[i]
            new_t_end_idx = t_end_idx_orig + num_inj
            new_s_end_idx = s_end_idx_orig + num_inj * 2 if s_end_idx_orig > t_end_idx_orig else t_end_idx_orig + num_inj
            
            # 使用新边界进行切片
            pert_time = pure_perturbation[i, :new_t_end_idx + 1]
            base_time = original_tor_windows[i, :t_end_idx_orig + 1]
            if new_s_end_idx > new_t_end_idx:
                pert_size = pure_perturbation[i, new_t_end_idx + 1 : new_s_end_idx + 1]
                base_size = original_tor_windows[i, t_end_idx_orig + 1 : s_end_idx_orig + 1]
            else:
                pert_size = base_size = torch.tensor([], device=device)
            
            epsilon = 1e-9
            session_time_ratios.append((torch.linalg.norm(pert_time) / (torch.linalg.norm(base_time))).item())
            session_size_ratios.append((torch.linalg.norm(pert_size) / (torch.linalg.norm(base_size))).item())
        
        all_time_ratios.append(np.nanmean(session_time_ratios) if session_time_ratios else 0)
        all_size_ratios.append(np.nanmean(session_size_ratios) if session_size_ratios else 0)

        # --- 还原、重切分与评估 ---
        reverted_session = reconstruct_single_session(adv_tor_windows, original_session, boundaries, time_indices, size_indices)
        repartitioned_adv_windows, _, _, _ = partition_single_session(reverted_session, args.delta, args.win_size, args.n_wins, args.tor_len, device)
        
        with torch.no_grad():
            orig_tor_emb = anchor_model(original_tor_windows)
            exit_emb = pandn_model(original_exit_windows)
            adv_tor_emb = anchor_model(repartitioned_adv_windows)
            all_original_sims.append(F.cosine_similarity(orig_tor_emb, exit_emb).mean().item())
            all_attacked_sims.append(F.cosine_similarity(adv_tor_emb, exit_emb).mean().item())

        total_repartitioned_adv_windows.append(repartitioned_adv_windows.cpu().numpy())
        total_original_tor_windows.append(original_tor_windows.cpu().numpy())
        total_original_exit_windows.append(original_exit_windows.cpu().numpy())

    avg_orig_sim, avg_adv_sim = np.mean(all_original_sims), np.mean(all_attacked_sims)
    avg_time_ratio, avg_size_ratio = np.nanmean(all_time_ratios), np.nanmean(all_size_ratios)

    print("\n--- BLANKET on DeepCoffea: 评估结果 ---")
    print(f"攻击样本数: {args.num_samples}")
    print(f"攻击前平均相似度: {avg_orig_sim:.4f}")
    print(f"攻击后平均相似度:  {avg_adv_sim:.4f}")
    print(f"平均L2扰动比率 (Time): {avg_time_ratio:.4f}")
    print(f"平均L2扰动比率 (Size): {avg_size_ratio:.4f}")
    
    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    result_fname = f"BLANKETdeepcoffea_time{avg_time_ratio:.4f}_size{avg_size_ratio:.4f}.p"
    with open(save_dir / result_fname, "wb") as fp:
        pickle.dump({
            "adv_tor_windows": total_repartitioned_adv_windows, "original_tor_windows": total_original_tor_windows,
            "original_exit_windows": total_original_exit_windows, "original_sims": all_original_sims,
            "attacked_sims": all_attacked_sims, "time_ratios": all_time_ratios,
            "size_ratios": all_size_ratios, "args": vars(args)}, fp)
    print(f"结果已保存至: {save_dir / result_fname}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用修正后的L2范数计算方法评估BLANKET对DeepCoffea的攻击')
    # --- 路径参数 ---
    parser.add_argument('--target_model_path', type=str, default="target_model/deepcoffea/deepcoffea_d3_ws5_nw11_thr20_tl500_el800_nt1000_ap1e-01_es64_lr1e-03_mep100000_bs256/")
    parser.add_argument('--blanket_model_path', type=str, default= "baseline/BLANKET/model_blanketdeepcoffea/mid0.2500_sigma0.5000_numadd50_alladdsize40.00/blanket_deepcoffea_epoch4_batch38_loss0.61.pth")
    parser.add_argument('--data_path', type=str, default="target_model/deepcoffea/dataset/CrawlE_Proc/")
    parser.add_argument('--save_dir', type=str, default="baseline/BLANKET/")
    # --- 评估和数据参数 ---
    parser.add_argument('--device', type=str, default="cuda:1")
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument("--delta", default=3, type=int)
    parser.add_argument("--win_size", default=5, type=int)
    parser.add_argument("--n_wins", default=11, type=int)
    parser.add_argument("--threshold", default=20, type=int)
    parser.add_argument("--tor_len", default=500, type=int)
    parser.add_argument("--exit_len", default=800, type=int)
    parser.add_argument("--n_test", default=1000, type=int)
    parser.add_argument("--emb_size", default=64, type=int)
    # --- BLANKET攻击超参数 ---
    parser.add_argument('--mid', type=float, default=0.25)
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--to-add', type=int, default=50)
    parser.add_argument('--size-padding-total-kb', type=float, default=40.0)
    parser.add_argument('--size-padding-packet-bytes', type=float, default=256)
    
    args = parser.parse_args()
    main(args)