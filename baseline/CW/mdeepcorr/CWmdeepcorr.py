import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import os
import argparse
from tqdm import tqdm
import pathlib
from target_model.Deepcorr700 import Model as Net

# ------------------------------------------------------------
# C&W L2 攻击函数 (加权L2损失版)
# ------------------------------------------------------------
def deepcorr_cw_attack(model, original_sample, device, attack_mask,
                         time_channels, size_channels, # 新增
                         binary_search_steps, num_iter_optim,
                         confidence, initial_c, lr,
                         time_l2_weight): # 新增
    """
    执行C&W L2攻击，但对time和size特征的L2损失进行加权处理。
    """
    model.eval()
    
    best_output = float('inf')
    best_adv_sample = original_sample.clone()

    lower_bound_c = 0.0
    upper_bound_c = initial_c
    flag = False
    for step in range(binary_search_steps):
        c = (lower_bound_c + upper_bound_c) / 2
        
        delta = torch.zeros_like(original_sample, requires_grad=True, device=device)
        optimizer = optim.Adam([delta], lr=lr)
        
        for i in range(num_iter_optim):
            optimizer.zero_grad()
            
            adv_sample = original_sample + delta * attack_mask
            adv_sample = torch.clamp(adv_sample, min=0)
            
            output = model(adv_sample, dropout=0.0)
            
            # --- L2损失计算修改点 ---
            # 1. 计算总的、掩码后的扰动
            perturbation = (adv_sample - original_sample) * attack_mask
            # 2. 将扰动按信道分开
            pert_time = perturbation[:, :, time_channels, :]
            pert_size = perturbation[:, :, size_channels, :]
            # 3. 分别计算L2损失的平方
            l2_dist_sq_time = torch.linalg.norm(pert_time,ord=2, dim=-1).mean()
            l2_dist_sq_size = torch.linalg.norm(pert_size,ord=2, dim=-1).mean()
            origin_time = original_sample[:, :, time_channels, :]
            origin_size = original_sample[:, :, size_channels, :]   
            origin_time_l2 = torch.linalg.norm(origin_time, ord=2, dim=-1).mean()
            origin_size_l2 = torch.linalg.norm(origin_size, ord=2, dim=-1).mean()
            time_ratio = l2_dist_sq_time / (origin_time_l2 )
            size_ratio = l2_dist_sq_size / (origin_size_l2 )
            # 4. 使用权重将它们加权求和
            l2_dist_sq = (time_l2_weight * time_ratio) + size_ratio
            # --- 修改结束 ---
            # loss_cls = torch.maximum(torch.tensor(0.0).to(device), output + confidence)
            loss_cls = torch.sigmoid(output)
            # loss_cls = torch.maximum(torch.tensor(0.0).to(device), loss_cls -0.0005)
            total_loss = l2_dist_sq + c * loss_cls
            
            total_loss.backward()
            optimizer.step()

        with torch.no_grad():
            final_adv_sample = torch.clamp(original_sample + delta * attack_mask, min=0)
            final_output = model(final_adv_sample, dropout=0.0)
            torch.sigmoid(final_output, out=final_output)
        perturbation = (adv_sample - original_sample) * attack_mask

        pert_time = perturbation[:, :, time_channels, :]
        pert_size = perturbation[:, :, size_channels, :]
        l2_dist_sq_time = torch.linalg.norm(pert_time,ord=2, dim=-1)
        l2_dist_sq_size = torch.linalg.norm(pert_size,ord=2, dim=-1)
        
        origin_time = original_sample[:, :, time_channels, :]
        origin_size = original_sample[:, :, size_channels, :]   
        origin_time_l2 = torch.linalg.norm(origin_time, ord=2, dim=-1)
        origin_size_l2 = torch.linalg.norm(origin_size, ord=2, dim=-1)
        
        time_ratio = (l2_dist_sq_time / (origin_time_l2 )).mean()
        size_ratio = (l2_dist_sq_size / (origin_size_l2 )).mean()
        print(f"Step {step+1}/{binary_search_steps}, c: {c:.4f}, "
              f"Time L2 Ratio: {time_ratio.item():.4f}, Size L2 Ratio: {size_ratio.item():.4f}, "
              f"Output: {final_output.item():.6f}") 
        
        if time_ratio < 0.15 and size_ratio < 0.15 : 
            if final_output < best_output:
                best_output = final_output
                best_adv_sample = final_adv_sample
                final_time_ratio = time_ratio
                final_size_ratio = size_ratio
            lower_bound_c = c
            flag = True
        else: 
            upper_bound_c = c
            if not flag:
                if time_ratio > 0.15 :
                    ratio = l2_dist_sq_time / (origin_time_l2 )
                    scaling_factor = torch.ones_like(ratio)
                    exceed = ratio > 0.15
                    scaling_factor[exceed] = 0.15 / ratio[exceed]
                    perturbation[:, :, time_channels, :] *= scaling_factor.unsqueeze(-1)
                if size_ratio > 0.15 :
                    ratio = l2_dist_sq_size / (origin_size_l2 )
                    scaling_factor = torch.ones_like(ratio)
                    exceed = ratio > 0.15
                    scaling_factor[exceed] = 0.15 / ratio[exceed]
                    perturbation[:, :, size_channels, :] *= scaling_factor.unsqueeze(-1)
                # 更新样本到边界状态
                best_adv_sample = original_sample + perturbation * attack_mask
    try:    
        print(f"Best similarity achieved: {best_output.item():.6f}")
        print(f"Best perturbation ratios - Time: {final_time_ratio:.4f}, Size: {final_size_ratio:.4f}")
    except Exception as e:
        print(f"未攻击成功……")
    #     final_adv_sample = torch.clamp(input=final_adv_sample,min=final_adv_sample,max=1.15*original_sample)
    #     best_adv_sample = final_adv_sample
        # if final_output.item() < 0.001:
        #     if time_ratio+size_ratio < best_l2:
        #         best_l2 = time_ratio+size_ratio
        #         best_adv_sample = final_adv_sample
        #     upper_bound_c = c
        # else:
        #     lower_bound_c = c
            
    return best_adv_sample.detach()

# ------------------------------------------------------------
# 3. 数据生成函数 
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
    print(f"Using device: {device}")

    model = Net().to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Successfully loaded pre-trained model from {args.model_path}.")
    except Exception as e:
        print(f"Error loading model: {e}.")
        return
    model.eval()

    print("Loading and preparing test dataset...")
    dataset = []
    all_runs={'8872':'192.168.122.117','8802':'192.168.122.117','8873':'192.168.122.67','8803':'192.168.122.67',
             '8874':'192.168.122.113','8804':'192.168.122.113','8875':'192.168.122.120',
            '8876':'192.168.122.30','8877':'192.168.122.208','8878':'192.168.122.58'}
    for name in all_runs:
        try:
            dataset_path = os.path.join(args.data_folder, f'{name}_tordata300.pickle')
            with open(dataset_path, 'rb') as f: dataset += pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: Dataset file {dataset_path} not found, skipping.")
    
    try:
        test_index_path = os.path.join(args.index_folder, 'test_index.pickle')
        with open(test_index_path, 'rb') as f: test_index = pickle.load(f)[:args.num_samples]
    except FileNotFoundError:
        print(f"Error: Test index file {test_index_path} not found. Exiting.")
        return
    
    test_samples = generate_positive_samples_only(dataset, test_index)

    time_channels = [0, 3]
    size_channels = [4, 7]
    attack_channels = time_channels + size_channels
    
    attack_mask = torch.zeros_like(test_samples[0]).unsqueeze(0).to(device)
    attack_mask[:, :, attack_channels, :] = 1

    total_adv_samples, total_original_samples = [], []
    total_time_ratios, total_size_ratios = [], []
    successful_attacks = 0

    print(f"Starting C&W L2 attack on {len(test_samples)} samples...")
    for i in tqdm(range(len(test_samples)), desc="Attacking Samples"):
        original_sample = test_samples[i].unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            initial_logit = model(original_sample, dropout=0.0).item()
        
        if initial_logit <= 0:
            continue

        # --- 调用修改后的C&W攻击函数 ---
        adv_sample = deepcorr_cw_attack(
            model=model,
            original_sample=original_sample,
            device=device,
            attack_mask=attack_mask,
            time_channels=time_channels,
            size_channels=size_channels,
            binary_search_steps=args.binary_search_steps,
            num_iter_optim=args.num_iter,
            confidence=args.confidence,
            initial_c=args.initial_c,
            lr=args.lr,
            time_l2_weight=args.time_l2_weight
        )
        
        with torch.no_grad():
            final_pert = adv_sample - original_sample
            
            original_time_features = original_sample[:, :, time_channels, :]
            original_size_features = original_sample[:, :, size_channels, :]
            final_pert_time = final_pert[:, :, time_channels, :]
            final_pert_size = final_pert[:, :, size_channels, :]

            time_ratio = (torch.linalg.norm(final_pert_time,ord=2,dim=-1) / (torch.linalg.norm(original_time_features,ord=2,dim=-1))).mean().item()
            size_ratio = (torch.linalg.norm(final_pert_size,ord=2,dim=-1) / (torch.linalg.norm(original_size_features,ord=2,dim=-1))).mean().item()
        
        total_adv_samples.append(adv_sample.cpu())
        total_original_samples.append(original_sample.cpu())
        total_time_ratios.append(time_ratio)
        total_size_ratios.append(size_ratio)

    print("\n--- Attack Execution Complete ---")
    avg_time_ratio = np.mean(total_time_ratios)
    avg_size_ratio = np.mean(total_size_ratios)

    print(f"Average Time Feature L2 Perturbation Ratio: {avg_time_ratio:.4f}")
    print(f"Average Size Feature L2 Perturbation Ratio: {avg_size_ratio:.4f}")

    result_fpath = pathlib.Path(f'baseline/CW/CWdeepcorr_advsamples_time{avg_time_ratio:.4f}_size{avg_size_ratio:.4f}.p')
    result_fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(result_fpath, "wb") as fp:
        results = {
            "total_adv_samples": total_adv_samples,
            "total_original_samples": total_original_samples,
            "avg_time_l2_ratio": avg_time_ratio,
            "avg_size_l2_ratio": avg_size_ratio,
        }
        pickle.dump(results, fp)
    print(f"Adversarial samples and results saved to: {result_fpath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Weighted C&W L2 attack on DeepCorr")
    # Paths
    parser.add_argument("--model_path", type=str, default="target_model/deepcorr/deepcorr700/tor700_199_epoch11_acc0.88.pth", help="Path to the pre-trained DeepCorr model.")
    parser.add_argument("--data_folder", type=str, default="target_model/deepcorr/dataset/", help="Path to the folder containing .pickle dataset files.")
    parser.add_argument("--index_folder", type=str, default="target_model/mdeepcorr/", help="Path to the folder containing test_index.pickle.")
    parser.add_argument("--device", type=str, default="cuda:1", help="Computation device (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to attack.")
    
    # C&W Attack Hyperparameters
    parser.add_argument("--confidence", type=float, default=0, help="Confidence parameter (kappa) for C&W loss.")
    parser.add_argument("--initial_c", type=float, default=10, help="Initial value for the constant 'c' in the binary search.")
    parser.add_argument("--lr", type=float, default=1, help="Learning rate for the Adam optimizer in the attack.")
    parser.add_argument("--binary_search_steps", type=int, default=15, help="Number of steps for the binary search over 'c'.")
    parser.add_argument("--num_iter", type=int, default=100, help="Number of optimization iterations for each binary search step.")
    parser.add_argument("--time_l2_weight", type=float, default=0.0001, help="Weight for the L2 loss of time features. Use a value < 1.0 if time perturbations are too large.")
    
    args = parser.parse_args()
    main(args)
    
    