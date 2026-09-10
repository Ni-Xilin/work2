# -*- coding: utf-8 -*-

"""
本脚本用于评估预训练的BLANKET攻击模型对DeepCorr模型的攻击效果。
核心功能是应用BLANKET扰动，并使用“对齐”方法精确计算包含数据包注入在内的
L2范数扰动大小。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
import argparse
from tqdm import tqdm
import pathlib
from target_model.Deepcorr700 import Model as Net  # 假设Net是DeepCorr模型的定义

# --- BLANKET攻击中使用的函数和生成器模型 ---

def transfer_adv(inp, eps, mid, just_positive):
    """对时间扰动进行标准化，控制其均值和标准差。"""
    x = inp
    if just_positive == 1:
        x = F.relu(x)
    res = ((x - torch.clamp(x.mean(dim=1, keepdim=True) - mid, min=0) - torch.clamp(x.mean(dim=1, keepdim=True) + mid, max=0)))
    res_multi = (torch.clamp(x.std(dim=1, keepdim=True), max=eps) / (x.std(dim=1, keepdim=True)) + 1e-9)
    res = res * res_multi
    return res

class TIMENOISER(nn.Module):
    """生成时间扰动的模型。"""
    def __init__(self, inp, just_positive_flag):
        super(TIMENOISER, self).__init__()
        self.inp = inp
        self.just_positive = just_positive_flag
        self.independent = nn.Sequential(nn.Linear(inp, 1000), nn.ReLU(), nn.Linear(1000, inp))

    def forward(self, z_input, eps, mid, outsize=700):
        ind = self.independent(z_input)
        res = transfer_adv(ind, eps, mid, self.just_positive)
        res = res.view(-1, 1, self.inp)
        if self.inp < outsize:
            z = torch.zeros_like(res)
            res = torch.cat([res] + [z] * 5, dim=2)[:, :, :outsize]
        z = torch.zeros_like(res)
        x = torch.stack([res, z, z, res, z, z, z, z], dim=2)
        return x

def generate_perturbation(change_points, size=700):
    """
    生成一个排列图谱 (permutation map)。
    这个图谱定义了如何移动旧数据包，为新注入的数据包腾出空间。
    """
    start = size - len(change_points)
    pert = []
    passed = 0
    for ind in range(size):
        if ind in change_points:
            pert.append(start)
            start += 1
            passed += 1
        else:
            pert.append(ind - passed)
    return pert

class PacketWithSizeFunction(torch.autograd.Function):
    """
    这是一个自定义的PyTorch函数，用于执行数据包注入和移位操作。
    它不是一个可训练的层，而是一个固定的、不可导的操作。
    """
    @staticmethod
    def forward(ctx, noise, sizes, inp, num, perts_map):
        num = int(num)
        if num == 0:
            return inp
        
        # 确定新注入数据包在最终向量中的位置
        tops = torch.argsort(noise, descending=False)[:num]
        
        # 使用预先计算好的排列图谱
        perts = torch.tensor(perts_map, device=inp.device)
        
        output = inp.clone()
        
        # 设定注入数据包的特征值
        time_adv_value = torch.ones_like(noise[tops]) * 0.001
        size_adv_value = 0.595 * ((sizes[tops] > 0).float() + 1)

        # 核心步骤：根据排列图谱移动所有通道的数据
        for ch in range(8):
            output[:, :, ch, :] = output[:, :, ch, perts]
        
        # 在`tops`位置注入新数据包的特征
        output[:, :, 0, tops] = time_adv_value
        output[:, :, 3, tops] = time_adv_value
        output[:, :, 4, tops] = size_adv_value
        output[:, :, 7, tops] = size_adv_value
            
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 在评估阶段，我们不需要反向传播，所以返回None
        return None, None, None, None, None

class SizePaddingRemapping(torch.autograd.Function):
    """自定义函数，用于执行大小填充操作。"""
    @staticmethod
    def forward(ctx, raw_perturbation, data_in, total_overhead_bytes, per_packet_overhead_bytes):
        data_out = data_in.clone()
        if total_overhead_bytes == 0:
            return data_out
        indices_to_perturb = torch.argsort(raw_perturbation, descending=True)
        remaining_overhead = total_overhead_bytes
        for i in indices_to_perturb:
            if remaining_overhead <= 0:
                break
            delta = min(per_packet_overhead_bytes, remaining_overhead)
            remaining_overhead -= delta
            # 将填充量加到Tor侧的大小通道 (4和7)
            data_out[:, 0, 4, i] += delta / 1000.0
            data_out[:, 0, 7, i] += delta / 1000.0
        return data_out

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None

# 为自定义函数创建别名，方便调用
decider = PacketWithSizeFunction.apply
size_pad_remapper = SizePaddingRemapping.apply

class ADDNOISER(nn.Module):
    """生成注入数据包的位置和大小扰动的模型。"""
    def __init__(self, inp, device):
        super(ADDNOISER, self).__init__()
        self.inp = inp
        self.z = torch.FloatTensor(size=(1, inp)).to(device)
        self.independent_where = nn.Sequential(nn.Linear(inp, 1000), nn.ReLU(), nn.Linear(1000, inp))
        self.independent_size = nn.Sequential(nn.Linear(inp, 1000), nn.ReLU(), nn.Linear(1000, inp))

    def forward(self, outsize=700):
        self.z.uniform_(-0, 0.5)
        ind_where = self.independent_where(self.z)
        ind_size = self.independent_size(self.z)
        if self.inp < outsize:
            z = torch.zeros_like(ind_where)
            ind_where = torch.cat([ind_where] + [z]*5, dim=1)[:, :outsize]
            ind_size = torch.cat([ind_size] + [z]*5, dim=1)[:, :outsize]
        return ind_where.view(-1), ind_size.view(-1)

class SIZEPADNOISER(nn.Module):
    """生成大小填充扰动的模型。"""
    def __init__(self, inp, device):
        super(SIZEPADNOISER, self).__init__()
        self.inp = inp
        self.z = torch.FloatTensor(size=(1, inp)).to(device)
        self.generator = nn.Sequential(nn.Linear(inp, 1000), nn.ReLU(), nn.Linear(1000, inp))

    def forward(self, outsize=700):
        self.z.uniform_(-0, 0.5)
        raw_pert = self.generator(self.z)
        if self.inp < outsize:
            z = torch.zeros_like(raw_pert)
            raw_pert = torch.cat([raw_pert] + [z]*5, dim=1)
        return raw_pert[:, :outsize].view(-1)


# ------------------------------------------------------------
# 2. 攻击应用与数据加载函数
# ------------------------------------------------------------

def pad_or_truncate(data_list, target_length):
    """
    将列表填充零或截断到目标长度。
    """
    current_length = len(data_list)
    if current_length >= target_length:
        return data_list[:target_length]
    else:
        padding_needed = target_length - current_length
        return data_list + [0.0] * padding_needed
    
    
def generate_positive_samples_only(dataset, test_index, flow_size=700):
    l2s_test = torch.zeros((len(test_index), 1, 8, flow_size)) 
    labels_test = torch.ones(len(test_index))
    for index, i in enumerate(test_index):
        l2s_test[index, 0, 0,:] = torch.tensor(pad_or_truncate(dataset[i]['here'][0]['<-'][:flow_size],700)) * 1000.0  
        l2s_test[index, 0, 1,:] = torch.tensor(pad_or_truncate(dataset[i]['there'][0]['->'][:flow_size],700)) * 1000.0  
        l2s_test[index, 0, 2,:] = torch.tensor(pad_or_truncate(dataset[i]['there'][0]['<-'][:flow_size],700)) * 1000.0  
        l2s_test[index, 0, 3,:] = torch.tensor(pad_or_truncate(dataset[i]['here'][0]['->'][:flow_size],700)) * 1000.0   

        l2s_test[index, 0, 4,:] = torch.tensor(pad_or_truncate(dataset[i]['here'][1]['<-'][:flow_size],700)) / 1000.0 
        l2s_test[index, 0, 5,:] = torch.tensor(pad_or_truncate(dataset[i]['there'][1]['->'][:flow_size],700)) / 1000.0
        l2s_test[index, 0, 6,:] = torch.tensor(pad_or_truncate(dataset[i]['there'][1]['<-'][:flow_size],700)) / 1000.0
        l2s_test[index, 0, 7,:] = torch.tensor(pad_or_truncate(dataset[i]['here'][1]['->'][:flow_size],700)) / 1000.0
    
    return l2s_test

def apply_blanket_attack(original_sample, timenois, addnois, sizepadnois, device,
                         num_to_add, sigma, mid, inpsize,
                         size_padding_total_kb, size_padding_packet_bytes):
    """
    应用BLANKET扰动，并返回最终的扰动后样本(X')和对齐后的原始样本(X_aligned)。
    这是实现正确L2范数计算的核心函数。
    """
    timenois.eval()
    addnois.eval()
    sizepadnois.eval()

    with torch.no_grad():
        # --- 步骤 1: 数据包注入 ---
        # 从ADDNOISER获取注入的位置和大小的原始扰动
        where, sizes = addnois()
        
        # 确定注入位置(tops)和用于移位的排列图谱(perts_map)
        tops = torch.argsort(where, descending=False)[:int(num_to_add)]
        perts_map = generate_perturbation(tops.cpu().numpy(), size=inpsize)
        
        # 应用注入和移位操作，得到基础的扰动后样本
        data_adv = decider(where, sizes, original_sample, num_to_add, perts_map)
        
        # --- 步骤 2: 创建对齐后的原始样本 (X_aligned) ---
        # 目标：创建一个与data_adv具有相同维度和对齐方式的原始样本版本
        aligned_original_sample = original_sample.clone()
        # 1. 应用与上面完全相同的排列图谱，移动原始数据包
        perts_tensor = torch.tensor(perts_map, device=device)
        for ch in range(8):
            aligned_original_sample[:, :, ch, :] = aligned_original_sample[:, :, ch, perts_tensor]
        # 2. 将新注入数据包的位置清零，因为这些位置在原始样本中不存在
        aligned_original_sample[:, :, :, tops] = 0.0

        # --- 步骤 3: 应用大小填充和时间扰动 ---
        # 这两步是元素级别的操作，不会改变对齐方式，因此在data_adv上继续进行
        
        # 大小填充
        raw_size_pad_pert = sizepadnois()
        total_overhead_bytes = size_padding_total_kb * 1024
        data_adv_padded = size_pad_remapper(raw_size_pad_pert, data_adv, total_overhead_bytes, size_padding_packet_bytes)
        
        # 时间扰动
        z_time = torch.ones(1, inpsize).to(device).uniform_(-2, 2)
        adv_time = timenois(z_time, sigma, mid)
        final_perturbed_data = torch.clamp(data_adv_padded + adv_time, min=0)

    # 返回最终的扰动后样本(X')和对齐后的原始样本(X_aligned)
    return final_perturbed_data.detach(), aligned_original_sample.detach()

# ------------------------------------------------------------
# 3. 主评估函数
# ------------------------------------------------------------
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # --- 加载目标模型和BLANKET生成器 ---
    target_model = Net().to(device)
    try:
        target_model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"目标模型DeepCorr加载成功: {args.model_path}")
    except Exception as e:
        print(f"加载目标模型失败: {e}")
        return
    target_model.eval()

    timenois = TIMENOISER(args.input_size, args.justpos).to(device)
    addnois = ADDNOISER(args.input_size, device).to(device)
    sizepadnois = SIZEPADNOISER(args.input_size, device).to(device)
    try:
        blanket_models = torch.load(args.blanket_model_path, map_location=device)
        timenois.load_state_dict(blanket_models['time_model'])
        addnois.load_state_dict(blanket_models['add_model'])
        sizepadnois.load_state_dict(blanket_models['sizepad_model'])
        print(f"BLANKET攻击模型加载成功: {args.blanket_model_path}")
    except Exception as e:
        print(f"加载BLANKET模型失败: {e}")
        return
    
    # --- 加载测试数据 ---
    print("加载并准备测试数据集...")
    dataset = []
    all_runs={'8872':'192.168.122.117','8802':'192.168.122.117','8873':'192.168.122.67','8803':'192.168.122.67',
             '8874':'192.168.122.113','8804':'192.168.122.113','8875':'192.168.122.120',
            '8876':'192.168.122.30','8877':'192.168.122.208','8878':'192.168.122.58'}
    for name in all_runs:
        try:
            dataset_path = os.path.join(args.data_folder, f'{name}_tordata300.pickle')
            with open(dataset_path, 'rb') as f: dataset += pickle.load(f)
        except FileNotFoundError:
            print(f"警告: 数据集文件 {dataset_path} 未找到, 已跳过。")
    try:
        test_index_path = os.path.join(args.index_folder, 'test_index.pickle')
        with open(test_index_path, 'rb') as f: test_index = pickle.load(f)[:args.num_samples]
    except FileNotFoundError:
        print(f"错误: 测试索引文件 {test_index_path} 未找到。程序退出。")
        return
    
    test_samples = generate_positive_samples_only(dataset, test_index)

    time_channels = [0, 3] # Tor侧的时间特征通道
    size_channels = [4, 7] # Tor侧的大小特征通道

    total_adv_samples_list, total_original_samples_list = [], []
    total_time_ratios, total_size_ratios = [], []

    print(f"开始对 {len(test_samples)} 个样本进行BLANKET攻击...")
    for i in tqdm(range(len(test_samples)), desc="攻击样本中"):
        original_sample = test_samples[i].unsqueeze(0).float().to(device)
        
        # --- 应用BLANKET攻击，获取扰动后样本(X')和对齐后原始样本(X_aligned) ---
        adv_sample, aligned_original_sample = apply_blanket_attack(
            original_sample=original_sample,
            timenois=timenois, addnois=addnois, sizepadnois=sizepadnois, device=device,
            num_to_add=args.to_add, sigma=args.sigma, mid=args.mid, inpsize=args.input_size,
            size_padding_total_kb=args.size_padding_total_kb,
            size_padding_packet_bytes=args.size_padding_packet_bytes
        )
        
        # --- 基于对齐的向量，计算“纯粹”的扰动(δ)和L2范数 ---
        with torch.no_grad():
            # 核心步骤: δ = X' - X_aligned
            pure_perturbation = adv_sample - aligned_original_sample
            
            # 按特征类型（时间/大小）分离纯粹扰动向量
            pert_time = pure_perturbation[:, :, time_channels, :]
            pert_size = pure_perturbation[:, :, size_channels, :]
            
            # 按特征类型分离作为比较基准的“对齐后”原始向量
            base_time = aligned_original_sample[:, :, time_channels, :]
            base_size = aligned_original_sample[:, :, size_channels, :]

            # 分别计算扰动和基准的L2范数
            norm_pert_time = torch.linalg.norm(pert_time.flatten())
            norm_pert_size = torch.linalg.norm(pert_size.flatten())
            norm_base_time = torch.linalg.norm(base_time.flatten())
            norm_base_size = torch.linalg.norm(base_size.flatten())

            # 计算L2范数比率
            time_ratio = (norm_pert_time / (norm_base_time + 1e-12)).item()
            size_ratio = (norm_pert_size / (norm_base_size + 1e-12)).item()

        # 存储结果
        total_adv_samples_list.append(adv_sample.cpu())
        total_original_samples_list.append(original_sample.cpu()) # 存储未经任何修改的原始样本
        total_time_ratios.append(time_ratio)
        total_size_ratios.append(size_ratio)

    print("\n--- 攻击执行完毕 ---")
    avg_time_ratio = np.mean(total_time_ratios) if total_time_ratios else 0
    avg_size_ratio = np.mean(total_size_ratios) if total_size_ratios else 0

    print(f"平均时间特征L2扰动比率 (已校正): {avg_time_ratio:.4f}")
    print(f"平均大小特征L2扰动比率 (已校正): {avg_size_ratio:.4f}")

    # --- 保存结果 ---
    result_fname = (f'BLANKETdeepcorr_time{avg_time_ratio:.4f}_size{avg_size_ratio:.4f}_'
                    f'add{int(args.to_add)}_pad{int(args.size_padding_total_kb)}KB_sigma{int(args.sigma)}.p')
    result_fpath = pathlib.Path(f'baseline/BLANKET/mdeepcorr/{result_fname}')
    result_fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(result_fpath, "wb") as fp:
        results = {
            "total_adv_samples": total_adv_samples_list,
            "total_original_samples": total_original_samples_list,
            "avg_time_l2_ratio": avg_time_ratio,
            "avg_size_l2_ratio": avg_size_ratio,
        }
        pickle.dump(results, fp)
    print(f"对抗样本及评估结果已保存至: {result_fpath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用修正后的L2范数计算方法评估BLANKET对DeepCorr的攻击")
    # 路径参数
    parser.add_argument("--model_path", type=str, default="target_model/deepcorr/deepcorr700/tor700_199_epoch11_acc0.88.pth", help="预训练的DeepCorr模型路径。")
    parser.add_argument("--blanket_model_path", type=str, default ="baseline/BLANKET/mdeepcorr/mid5_sigma50_numadd50_alladdsize100.00_reg0.00_inpsize700_just_positive1/model_epoch17_avgloss11.21", help="预训练的BLANKET生成器模型路径(.pth文件)。")
    parser.add_argument("--data_folder", type=str, default="target_model/deepcorr/dataset/", help="包含.pickle数据集文件的文件夹路径。")
    parser.add_argument("--index_folder", type=str, default="target_model/mdeepcorr/", help="包含test_index.pickle文件的文件夹路径。")
    
    # 评估配置参数
    parser.add_argument("--device", type=str, default="cuda:0", help="计算设备 (例如 'cuda:0' 或 'cpu')。")
    parser.add_argument("--num_samples", type=int, default=1000, help="要攻击的样本数量。")
    parser.add_argument('--input-size', type=int, default=700, help='生成器模型的输入特征向量长度。')
    
    # BLANKET攻击超参数 (应与训练时使用的参数匹配)
    parser.add_argument('--mid', type=float, default=5.0, help="时间延迟的均值('μ')。")
    parser.add_argument('--sigma', type=float, default=50.0, help="时间延迟的标准差('σ')。")
    parser.add_argument('--justpos', type=int, default=1, help='若为1，则只产生正延迟(推迟)；若为0，则允许延迟和提前。')
    parser.add_argument('--to-add', type=float, default=50.0, help='要注入的虚拟数据包数量。')
    parser.add_argument('--size-padding-total-kb', type=float, default=100.0, help="通过填充增加的总带宽开销(KB)('N')。")
    parser.add_argument('--size-padding-packet-bytes', type=float, default=512.0, help="单个数据包允许添加的最大填充字节数('n')。")
    
    args = parser.parse_args()
    main(args)