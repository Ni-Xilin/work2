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
from torch.utils.data import Dataset, DataLoader

# 导入 DeepCoffea 模型定义
from target_model.Deepcoffea import Model

# --- 为了可复现性，设置随机种子 ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ------------------------------------------------------------
# 1. 数据处理与自定义 Autograd 函数
# ------------------------------------------------------------
def partition_single_session(session, delta, win_size, n_wins, tor_len, device):
    win_size_ms = win_size * 1000
    delta_ms = delta * 1000
    offset = win_size_ms - delta_ms
    session_ipd = session[0, :]
    session_size = session[1, :]
    cumulative_time = session_ipd.abs().cumsum(dim=0)
    
    partitioned_data_single, time_indices_single, size_indices_single = [], [], []

    for wi in range(int(n_wins)):
        start_time, end_time = wi * offset, wi * offset + win_size_ms
        start_idx, end_idx = torch.searchsorted(cumulative_time, start_time).item(), torch.searchsorted(cumulative_time, end_time).item()
        
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
    
    return torch.stack(partitioned_data_single, dim=0), time_indices_single, size_indices_single

# --- 【新增】排列生成函数，用于实现“插入”效果 ---
def generate_perturbation(change_points, size):
    if isinstance(change_points, torch.Tensor):
        change_points = change_points.cpu().numpy()
    
    start = size - len(change_points)
    pert = [] 
    passed = 0 
    change_points_set = set(change_points.tolist())
    for ind in range(size):
        if ind in change_points_set:
            pert.append(start)
            start += 1
            passed += 1
        else:
            pert.append(ind - passed)
    return pert

class PacketInjectionFunction(torch.autograd.Function):
    """
    【最终修正版】为 DeepCoffea 的一维窗口格式适配 BLANKET 的数据包注入逻辑。
    此版本为“插入”式，先排列移位，再覆盖注入。
    """
    @staticmethod
    def forward(ctx, noise_where, noise_size, data_in, num_to_add, time_end_idx_batch, size_end_idx_batch):
        batch_size = data_in.shape[0]
        data_out = data_in.clone()
        
        for i in range(batch_size):
            if num_to_add == 0: continue
            
            time_end_idx = time_end_idx_batch[i]
            max_len = data_in.shape[1]
            original_valid_data_len = time_end_idx + 1
            safe_data_len = min(original_valid_data_len, max_len)
            
            if safe_data_len <= 0: continue

            # 1. 确定要替换/插入的位置 (tops)
            num_to_inject = int(min(num_to_add, safe_data_len))
            candidate_indices = torch.argsort(noise_where[i, :safe_data_len])[:num_to_inject]
            
            # --- 【核心修改】排列移位 + 覆盖注入 ---
            # 2. 生成排列图谱，只针对IPD（真实数据）部分
            perts_map = generate_perturbation(candidate_indices, size=safe_data_len)
            perts_tensor = torch.tensor(perts_map, device=data_in.device)

            # 3. 应用排列，将保留的包前移，被替换的包推到末尾
            data_out[i, :safe_data_len] = data_out[i, :safe_data_len][perts_tensor]

            # 4. 在腾出的位置上注入新数据
            size_indices = original_valid_data_len + candidate_indices
            mask = size_indices < max_len
            final_time_indices = candidate_indices[mask]
            final_size_indices = size_indices[mask]
            
            if len(final_time_indices) == 0: continue

            time_signs = torch.sign(data_out[i, final_time_indices]); time_signs[time_signs == 0] = 1
            size_signs = torch.sign(data_out[i, final_size_indices]); size_signs[size_signs == 0] = 1
            original_size_val = 0.595 * (torch.sigmoid(noise_size[i, candidate_indices]))
            final_size_val = original_size_val[mask]
            final_time_val = torch.full_like(final_size_val, 1.0)
            data_out[i, final_time_indices] = time_signs * final_time_val
            data_out[i, final_size_indices] = size_signs * final_size_val
 
        return data_out

    @staticmethod
    def backward(ctx, grad_output):
        grad_where = grad_output
        grad_size = grad_output
        grad_data_in = grad_output
        return grad_where, grad_size, grad_data_in, None, None, None

# ... (SizePaddingFunction, TIMENOISER, ADDNOISER, SIZEPADNOISER, TimingDiscriminator 类定义保持不变) ...

#<editor-fold desc="Collapsed Code">
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
                data_out[i, actual_idx] += (original_sign * (delta/1000.0))
        return data_out
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output, None, None, None, None

inject_packets = PacketInjectionFunction.apply
pad_sizes = SizePaddingFunction.apply

class TIMENOISER(nn.Module):
    def __init__(self, in_size):
        super(TIMENOISER, self).__init__()
        self.network = nn.Sequential(nn.Linear(in_size, 512), nn.ReLU(), nn.Linear(512, in_size))
    def forward(self, z, time_end_idx_batch, sigma, mid):
        raw_pert = self.network(z)
        res = raw_pert - raw_pert.mean(dim=-1, keepdim=True) + mid
        res_std = res.std(dim=-1, keepdim=True)
        res = res * (sigma / (res_std + 1e-9))
        final_pert = torch.zeros_like(raw_pert)
        for i in range(z.shape[0]):
            time_end_idx = time_end_idx_batch[i]
            if time_end_idx >= 0:
                safe_slice_end = min(time_end_idx + 1, final_pert.shape[1])
                final_pert[i, :safe_slice_end] = res[i, :safe_slice_end]
        return final_pert

class ADDNOISER(nn.Module):
    def __init__(self, in_size):
        super(ADDNOISER, self).__init__()
        self.where_net = nn.Sequential(nn.Linear(in_size, 512), nn.ReLU(), nn.Linear(512, in_size))
        self.size_net = nn.Sequential(nn.Linear(in_size, 512), nn.ReLU(), nn.Linear(512, in_size))
    def forward(self, z):
        return self.where_net(z), self.size_net(z)

class SIZEPADNOISER(nn.Module):
    def __init__(self, in_size):
        super(SIZEPADNOISER, self).__init__()
        self.network = nn.Sequential(nn.Linear(in_size, 512), nn.ReLU(), nn.Linear(512, in_size))
    def forward(self, z):
        return self.network(z)

class TimingDiscriminator(nn.Module):
    def __init__(self, in_size):
        super(TimingDiscriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_size, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1))
    def forward(self, x):
        return self.network(x)
#</editor-fold>

# ------------------------------------------------------------
# 3. 核心训练逻辑 (批处理版)
# ------------------------------------------------------------
def train_batch(batch_data, generators, optim_gen, target_models, gan_components, args, device):
    mega_batch_tor_wins, mega_batch_exit_wins, mega_batch_time_indices, mega_batch_size_indices = batch_data
    time_noiser, add_noiser, pad_noiser = generators
    anchor_model, pandn_model = target_models
    timing_disc, optim_disc, bce_loss = gan_components
    
    anchor_model.eval(); pandn_model.eval()

    batch_size = mega_batch_tor_wins.shape[0]
    z = torch.randn(batch_size, args.tor_len * 2, device=device)
    
    where, sizes = add_noiser(z)
    adv_win_injected = inject_packets(where, sizes, mega_batch_tor_wins, args.to_add, mega_batch_time_indices, mega_batch_size_indices)
    
    pad_noise = pad_noiser(z)
    adv_win_padded = pad_sizes(pad_noise, adv_win_injected, mega_batch_time_indices, mega_batch_size_indices, 
                               args.size_padding_total_kb * 1024, args.size_padding_packet_bytes)
    
    time_pert = time_noiser(z, mega_batch_time_indices, args.sigma, args.mid)
    
    original_sign = torch.sign(adv_win_padded)
    signed_pert = original_sign * torch.abs(time_pert)
    final_adv_win = adv_win_padded + signed_pert

    if args.use_gan:
        # (GAN 训练逻辑保持不变)
        pass

    optim_gen.zero_grad()
    
    adv_tor_emb = anchor_model(final_adv_win)
    with torch.no_grad():
        exit_emb = pandn_model(mega_batch_exit_wins)
    
    loss_similarity = F.cosine_similarity(adv_tor_emb, exit_emb).mean()

    if args.use_gan:
        # (GAN 损失计算保持不变)
        pass
    else:
        total_gen_loss = loss_similarity

    total_gen_loss.backward()
    optim_gen.step()
    
    return total_gen_loss.item()

# ------------------------------------------------------------
# 4. 主执行函数
# ------------------------------------------------------------
class DeepCoffeaSessionDataset(Dataset):
    """为多进程加载而设计。"""
    def __init__(self, train_wins, train_sessions, args):
        train_exit = np.transpose(train_wins['train_exit'], (1, 0, 2))
        self.train_exit = torch.FloatTensor(train_exit)
        self.num_samples = self.train_exit.shape[0]
        self.train_sessions_ipds = train_sessions['tor_ipds']
        self.train_sessions_sizes = train_sessions['tor_sizes']
        self.args = args

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        cpu_device = torch.device('cpu')
        session_ipd = torch.tensor(self.train_sessions_ipds[idx], dtype=torch.float32, device=cpu_device)
        session_size = torch.tensor(self.train_sessions_sizes[idx], dtype=torch.float32, device=cpu_device)
        original_session = torch.stack([session_ipd, session_size], dim=0)
        
        tor_windows, time_indices, size_indices = partition_single_session(
            original_session, self.args.delta, self.args.win_size, self.args.n_wins, self.args.tor_len, cpu_device)
        
        exit_windows = self.train_exit[idx]
        time_indices = torch.tensor(time_indices, dtype=torch.long)
        size_indices = torch.tensor(size_indices, dtype=torch.long)

        return tor_windows, exit_windows, time_indices, size_indices

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("正在加载预训练的 DeepCoffea 模型...")
    anchor_model = Model(emb_size=args.emb_size, input_size=args.tor_len * 2).to(device)
    pandn_model = Model(emb_size=args.emb_size, input_size=args.exit_len * 2).to(device)
    model_path = os.path.join(args.target_model_path, 'best_loss.pth')
    state_dict = torch.load(model_path, map_location=device)
    anchor_model.load_state_dict(state_dict['anchor_state_dict'])
    pandn_model.load_state_dict(state_dict['pandn_state_dict'])
    print("目标模型加载成功。")

    print("正在加载数据集...")
    data_filename = f"d{args.delta}_ws{args.win_size}_nw{args.n_wins}_thr{args.threshold}_tl{args.tor_len}_el{args.exit_len}_nt{args.n_test}"
    session_path = os.path.join(args.data_path, "filtered_and_partitioned", f"{data_filename}_train_session.npz")
    win_path = os.path.join(args.data_path, "filtered_and_partitioned", f"{data_filename}_train.npz")
    train_sessions = np.load(session_path, allow_pickle=True)
    train_wins = np.load(win_path, allow_pickle=True)
    train_dataset = DeepCoffeaSessionDataset(train_wins, train_sessions, args)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, drop_last=True)
    
    input_dim = args.tor_len * 2
    time_noiser = TIMENOISER(input_dim).to(device); add_noiser = ADDNOISER(input_dim).to(device); pad_noiser = SIZEPADNOISER(input_dim).to(device)
    gen_params = list(time_noiser.parameters()) + list(add_noiser.parameters()) + list(pad_noiser.parameters())
    optim_gen = optim.Adam(gen_params, lr=args.gen_lr)

    gan_components = (None, None, None)
    if args.use_gan:
        print("GAN 模式已启用。")
        timing_disc = TimingDiscriminator(input_dim).to(device)
        optim_disc = optim.Adam(timing_disc.parameters(), lr=args.disc_lr)
        bce_loss = nn.BCEWithLogitsLoss()
        gan_components = (timing_disc, optim_disc, bce_loss)

    print(f"开始 BLANKET 训练，共 {args.epochs} 个周期。")
    for epoch in range(args.epochs):
        print(f"\n--- 周期 {epoch+1}/{args.epochs} ---")
        time_noiser.train(); add_noiser.train(); pad_noiser.train()
        epoch_losses = []
        
        for i, batch in enumerate(train_loader):
            tor_windows, exit_windows, time_indices, size_indices = batch
            
            mega_batch_tor_wins = tor_windows.view(-1, args.tor_len * 2).to(device)
            mega_batch_exit_wins = exit_windows.view(-1, args.exit_len * 2).to(device)
            mega_batch_time_indices = time_indices.view(-1).tolist()
            mega_batch_size_indices = size_indices.view(-1).tolist()

            batch_data = (mega_batch_tor_wins, mega_batch_exit_wins, mega_batch_time_indices, mega_batch_size_indices)
            
            loss = train_batch(batch_data, (time_noiser, add_noiser, pad_noiser), optim_gen, 
                               (anchor_model, pandn_model), gan_components, args, device)
            epoch_losses.append(loss)
            print(f"批次 {i+1} 训练损失: {loss:.4f}")
            if  epoch >1 and loss<0.63:
                save_path = pathlib.Path('baseline/BLANKET/model_blanketdeepcoffea/mid%.4f_sigma%.4f_numadd%d_alladdsize%.2f/'%(args.mid,args.sigma,args.to_add,args.size_padding_total_kb))
                save_path.mkdir(parents=True, exist_ok=True)
                model_filename = save_path / f"blanket_deepcoffea_epoch{epoch+1}_batch{i+1}_loss{loss:.2f}.pth"
                torch.save({
                    'time_noiser_state_dict': time_noiser.state_dict(),
                    'add_noiser_state_dict': add_noiser.state_dict(),
                    'pad_noiser_state_dict': pad_noiser.state_dict(),
                    'optimizer_state_dict': optim_gen.state_dict(),
                }, model_filename)
                print(f"生成器模型已保存至 {model_filename}")
            if( i + 1) % 100== 0:
                break
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='在 DeepCoffea 上训练 BLANKET 攻击 (多进程版)')
    parser.add_argument('--target_model_path', type=str, default="target_model/deepcoffea/deepcoffea_d3_ws5_nw11_thr20_tl500_el800_nt1000_ap1e-01_es64_lr1e-03_mep100000_bs256/")
    parser.add_argument('--data_path', type=str, default="target_model/deepcoffea/dataset/CrawlE_Proc/")
    parser.add_argument('--save_dir', type=str, default="baseline/BLANKET/model_blanketdeepcoffea/")
    parser.add_argument('--device', type=str, default="cuda:2")
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument("--delta", default=3, type=int)
    parser.add_argument("--win_size", default=5, type=int)
    parser.add_argument("--n_wins", default=11, type=int)
    parser.add_argument("--threshold", default=20, type=int)
    parser.add_argument("--tor_len", default=500, type=int)
    parser.add_argument("--exit_len", default=800, type=int)
    parser.add_argument("--n_test", default=1000, type=int)
    parser.add_argument("--emb_size", default=64, type=int)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gen_lr', type=float, default=0.001)
    parser.add_argument('--mid', type=float, default=0.25)
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--to-add', type=int, default=50)
    parser.add_argument('--size-padding-total-kb', type=float, default=40.0)
    parser.add_argument('--size-padding-packet-bytes', type=float, default=256)
    parser.add_argument('--use_gan', type =bool,default= False)
    parser.add_argument('--disc_lr', type=float, default=0.0001)
    parser.add_argument('--gan_reg_weight', type=float, default=0.05)
    
    args = parser.parse_args()
    main(args)
#</editor-fold>