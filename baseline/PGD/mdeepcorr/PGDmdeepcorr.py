import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
import argparse
from tqdm import tqdm
import pathlib
from target_model.Deepcorr700 import Model as Net

# ------------------------------------------------------------
# 最终版核心攻击函数
# ------------------------------------------------------------
def deepcorr_final_custom_attack(model, original_sample, device,
                                 attack_mask,
                                 time_channels, size_channels,
                                 max_ratio_time, max_ratio_size,
                                 alpha_tensor,
                                 num_iter
                                 ):
    """
    一个完全定制化的PGD攻击, 满足所有指定要求。
    """
    model.eval()
    adv_sample = original_sample.clone().detach().to(device)

    # --- 要求4: 基于“平均通道范数”动态计算Epsilon ---
    with torch.no_grad():
        original_time_features = original_sample[:, :, time_channels, :]
        original_size_features = original_sample[:, :, size_channels, :]
        
        original_norm_time_avg = torch.linalg.norm(original_time_features, ord=2, dim=-1)
        original_norm_size_avg = torch.linalg.norm(original_size_features, ord=2, dim=-1)

        # epsilon_time_abs_avg = original_norm_time_avg * max_ratio_time
        # epsilon_size_abs_avg = original_norm_size_avg * max_ratio_size

    # 随机初始化 (保证为正,同时保证不对padding的0进行扰动）
    original_non_zero_mask = (original_sample > 0).float().to(device) 
    delta = torch.rand_like(adv_sample) * 0.001
    attack_mask = attack_mask* original_non_zero_mask
    delta = delta * attack_mask
    adv_sample = adv_sample + delta

    for i in range(num_iter):
        adv_sample.requires_grad = True
        model.zero_grad()
        output = model(adv_sample,dropout=0.0)
        loss = output
        loss.backward()
        
        if adv_sample.grad is None: break
            
        grad_data = adv_sample.grad.data
        
        # --- 保证数值只会变大 ---
        step = -alpha_tensor * grad_data * attack_mask
        positive_step_only = torch.clamp(step, min=0)
        adv_sample = adv_sample.data + positive_step_only

        # --- 使用投影约束并提前终止 ---
        total_perturbation = adv_sample - original_sample
        
        pert_time = total_perturbation[:, :, time_channels, :]
        norm_time_avg = torch.linalg.norm(pert_time, ord=2, dim=-1)
        
        pert_size = total_perturbation[:, :, size_channels, :]
        norm_size_avg = torch.linalg.norm(pert_size, ord=2, dim=-1)
        
        if ((norm_time_avg/original_norm_time_avg).mean()) > max_ratio_time or \
           ((norm_size_avg/original_norm_size_avg).mean()) > max_ratio_size :
            # 对超标的部分进行一次投影
            if ((norm_time_avg/original_norm_time_avg).mean()) > max_ratio_time :
                ratio = norm_time_avg/original_norm_time_avg
                scaling_factor = torch.ones_like(ratio)
                exceed = ratio > 0.15
                scaling_factor[exceed] = 0.15 / ratio[exceed]
                total_perturbation[:, :, time_channels, :] *= scaling_factor.unsqueeze(-1)
            
            if ((norm_size_avg/original_norm_size_avg).mean()) > max_ratio_size :
                ratio = norm_size_avg/original_norm_size_avg
                scaling_factor = torch.ones_like(ratio)
                exceed = ratio > 0.15
                scaling_factor[exceed] = 0.15 / ratio[exceed]
                total_perturbation[:, :, size_channels, :] *= scaling_factor.unsqueeze(-1)
            
            # 更新样本到边界状态
            adv_sample = original_sample + total_perturbation * attack_mask
            # 立即终止
            break
        else: # 如果没有超标，正常更新
            adv_sample = original_sample + total_perturbation * attack_mask
    
    # --- 要求3: 保留函数内的计算和打印 ---
    with torch.no_grad():
        final_pert = adv_sample - original_sample
        original_time_features = original_sample[:, :, time_channels, :]
        original_size_features = original_sample[:, :, size_channels, :]
        final_pert_time = final_pert[:, :, time_channels, :]
        final_pert_size = final_pert[:, :, size_channels, :]

        time_ratio = (torch.linalg.norm(final_pert_time, ord=2, dim=-1) / (torch.linalg.norm(original_time_features, ord=2, dim=-1) + 1e-12)).mean()
        size_ratio = (torch.linalg.norm(final_pert_size, ord=2, dim=-1) / (torch.linalg.norm(original_size_features, ord=2, dim=-1) + 1e-12)).mean()
        
        print(f"攻击完成于迭代 {i+1}/{num_iter}。 "
              f"最终时间比率(均): {time_ratio.item():.4f}, "
              f"最终大小比率(均): {size_ratio.item():.4f}")
    
    return adv_sample.detach(),time_ratio.item(), size_ratio.item()

# ------------------------------------------------------------
# 3. 数据生成函数 (保持不变)
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

# ------------------------------------------------------------
# 4. 主函数
# ------------------------------------------------------------
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = Net().to(device)
    try: model.load_state_dict(torch.load(args.model_path, map_location=device)); print(f"成功从 {args.model_path} 加载预训练模型。")
    except Exception as e: print(f"加载模型时出错: {e}。")
    model.eval()

    print("正在加载和准备测试数据集...")
    dataset = []
    all_runs={'8872':'192.168.122.117','8802':'192.168.122.117','8873':'192.168.122.67','8803':'192.168.122.67',
             '8874':'192.168.122.113','8804':'192.168.122.113','8875':'192.168.122.120',
            '8876':'192.168.122.30','8877':'192.168.122.208','8878':'192.168.122.58'}
    for name in all_runs:
        try:
            dataset_path = os.path.join(args.data_folder, f'{name}_tordata300.pickle')
            with open(dataset_path, 'rb') as f: dataset += pickle.load(f)
        except FileNotFoundError: print(f"警告: 数据集文件 {dataset_path} 未找到，已跳过。")
    try:
        test_index_path = os.path.join(args.index_folder, 'test_index.pickle')
        with open(test_index_path, 'rb') as f: test_index = pickle.load(f)[:args.num_samples]
    except FileNotFoundError: print(f"错误: 测试索引文件 {test_index_path} 未找到。程序无法继续。"); exit()
    test_samples = generate_positive_samples_only(dataset, test_index)

    time_channels = [0, 3]; size_channels = [4, 7]
    attack_channels = time_channels + size_channels
    attack_mask = torch.zeros_like(test_samples[0]).unsqueeze(0).to(device)
    attack_mask[:, :, attack_channels, :] = 1
    
    alpha_tensor = torch.zeros_like(test_samples[0]).unsqueeze(0).to(device)
    alpha_tensor[:, :, time_channels, :] = args.alpha_time
    alpha_tensor[:, :, size_channels, :] = args.alpha_size

    total_adv_samples, total_original_samples,total_time_ratios,total_size_ratios = [], [],[],[]
    print(f"开始对 {len(test_samples)} 个样本进行最终版 Custom PGD 攻击...")
    for i in tqdm(range(len(test_samples))):
        original_sample = test_samples[i].unsqueeze(0).float().to(device)
        
        # 调用重构后的新攻击函数
        adv_sample,time_ratio,size_ratio = deepcorr_final_custom_attack(
            model=model,
            original_sample=original_sample,
            device=device,
            attack_mask=attack_mask,
            time_channels=time_channels,
            size_channels=size_channels,
            max_ratio_time=args.max_ratio_time,
            max_ratio_size=args.max_ratio_size,
            alpha_tensor=alpha_tensor,
            num_iter=args.num_iter
        )
        total_adv_samples.append(adv_sample.cpu())
        total_original_samples.append(original_sample.cpu())
        total_time_ratios.append(time_ratio)
        total_size_ratios.append(size_ratio)
    
    print("\n--- 攻击执行完毕 ---")
    print(f"平均时间特征L2扰动比率: {np.mean(total_time_ratios):.4f}")
    print(f"平均大小特征L2扰动比率: {np.mean(total_size_ratios):.4f}")
    result_fpath = pathlib.Path(f'baseline/PGD/mdeepcorr/PGDmdeepcorr_advsamples_time{np.mean(total_time_ratios):.4f}_size{np.mean(total_size_ratios):.4f}.p')
    with open(result_fpath, "wb") as fp:
        results = {
            "total_adv_samples": total_adv_samples,
            "total_original_samples": total_original_samples,
            "time_l2_ratios": np.mean(total_time_ratios),
            "size_l2_ratios": np.mean(total_size_ratios),
        }
        pickle.dump(results, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Final Custom PGD attack on DeepCorr")
    parser.add_argument("--model_path", type=str, default="target_model/deepcorr/deepcorr700/tor700_199_epoch11_acc0.88.pth", help="预训练DeepCorr模型的路径")
    parser.add_argument("--data_folder", type=str, default="target_model/deepcorr/dataset/", help="包含.pickle数据集文件的文件夹路径")
    parser.add_argument("--index_folder", type=str, default="target_model/mdeepcorr", help="包含test_index.pickle文件的文件夹路径")
    parser.add_argument("--device", type=str, default="cuda:1", help="计算设备")
    parser.add_argument("--num_samples", type=int, default=1000, help="用于攻击的样本数量")
    
    parser.add_argument("--num_iter", type=int, default=100, help="PGD的最大迭代次数")
    
    parser.add_argument("--max_ratio_time", type=float, default=0.15, help="时间特征的最大L2扰动比率")
    parser.add_argument("--max_ratio_size", type=float, default=0.15, help="大小特征的最大L2扰动比率")
    parser.add_argument("--alpha_time", type=float, default=1000, help="时间特征的迭代步长")
    parser.add_argument("--alpha_size", type=float, default=1, help="大小特征的迭代步长")
    
    args = parser.parse_args()
    main(args)