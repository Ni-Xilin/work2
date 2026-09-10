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
from target_model.Deepcorr700 import Model as Net
# ------------------------------------------------------------
# 核心攻击函数：I-FGSM (支持分别为time和size设置扰动大小)
# ------------------------------------------------------------
def deepcorr_ifgsm_attack(model, original_sample, time_indices, size_indices, device,
                          epsilon_time=0.01, epsilon_size=0.001,  # 使用两个独立的epsilon
                          num_iter=100, max_l2_ratio=0.2):
    """
    对单个deepcorr样本执行I-FGSM攻击，并为time和size特征使用不同的扰动步长。
    """
    model.eval()
    
    non_zero_original_mask = (original_sample != 0).float().to(device)
    adv_sample = original_sample.clone().detach().to(device)
    original_sample = original_sample.to(device)
    adv_sample.requires_grad = True

    with torch.no_grad():
        original_time_segment = original_sample[:, :, time_indices, :]
        original_size_segment = original_sample[:, :, size_indices, :]
        original_time_norm = torch.linalg.norm(original_time_segment.float(),ord=2,dim = -1)
        original_size_norm = torch.linalg.norm(original_size_segment.float(),ord=2,dim = -1)
        
        absolute_max_l2_time = original_time_norm * max_l2_ratio
        absolute_max_l2_size = original_size_norm * max_l2_ratio

    for i in range(num_iter):
        model.zero_grad()
        output = model(adv_sample,dropout=0)
        loss = output
        loss.backward()
        grads = adv_sample.grad.data
        sign_grads = grads.sign()
        
        # ##############################################################

        
        # 创建一个全零的扰动张量
        perturbation = torch.zeros_like(grads)
        
        # 1. 对时间特征施加扰动
        # 创建一个只包含时间特征通道的掩码
        time_mask = torch.zeros_like(grads, dtype=torch.bool)
        time_mask[:, :, time_indices, :] = True
        # 只在梯度为负的时间特征位置，应用epsilon_time
        perturbation[(sign_grads == -1) & time_mask & (non_zero_original_mask.bool())] = epsilon_time

        # 2. 对大小特征施加扰动
        # 创建一个只包含大小特征通道的掩码
        size_mask = torch.zeros_like(grads, dtype=torch.bool)
        size_mask[:, :, size_indices, :] = True
        # 只在梯度为负的大小特征位置，应用epsilon_size
        perturbation[(sign_grads == -1) & size_mask & (non_zero_original_mask.bool())] = epsilon_size
        
        # ##############################################################

        adv_sample.data = adv_sample.data + perturbation
        
        total_perturbation = adv_sample.data - original_sample.data
        current_time_pert = total_perturbation[:, :, time_indices, :]
        current_size_pert = total_perturbation[:, :, size_indices, :]
        current_time_pert_norm = torch.linalg.norm(current_time_pert.float(),ord=2,dim = -1)
        current_size_pert_norm = torch.linalg.norm(current_size_pert.float(),ord=2,dim = -1)

        if ((current_time_pert_norm/original_time_norm).mean() > max_l2_ratio or \
            (current_size_pert_norm/original_size_norm).mean() > max_l2_ratio):
            adv_sample.data = adv_sample.data - perturbation
            break
    print(f"迭代 {i+1}") 
    return adv_sample.detach()

# ------------------------------------------------------------
# 3. 主函数
# ------------------------------------------------------------
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # --- 加载模型 ---
    model = Net().to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"成功从 {args.model_path} 加载预训练模型。")
    except Exception as e:
        print(f"加载模型时出错: {e}。将使用随机初始化的模型。")

    # --- 加载数据集 ---
    print("正在加载和准备测试数据集...")
    # 这里沿用AJSAMdeepcorr.py的数据加载逻辑
    # ... (如果您的数据加载方式不同，请在此处修改) ...
    dataset = []
    all_runs={'8872':'192.168.122.117','8802':'192.168.122.117','8873':'192.168.122.67','8803':'192.168.122.67',
             '8874':'192.168.122.113','8804':'192.168.122.113','8875':'192.168.122.120',
            '8876':'192.168.122.30','8877':'192.168.122.208','8878':'192.168.122.58'}
    for name in all_runs:
        try:
            dataset_path = os.path.join(args.data_folder, f'{name}_tordata300.pickle')
            with open(dataset_path, 'rb') as f:
                 dataset += pickle.load(f)
        except FileNotFoundError:
            print(f"警告: 数据集文件 {dataset_path} 未找到。")
            continue
    
    try:
        test_index_path = os.path.join(args.index_folder, 'test_index.pickle')
        with open(test_index_path, 'rb') as f:
            test_index = pickle.load(f)[:args.num_samples]
    except FileNotFoundError:
        print(f"错误: {test_index_path} 文件未找到。无法创建测试集。")
        exit()

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

    test_samples = generate_positive_samples_only(dataset, test_index)
    test_dataset = TensorDataset(test_samples.float())
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # --- 定义特征索引 ---
    # 根据deepcorr的数据格式，0-3是时间相关，4-7是大小相关
    # 我们只攻击"here"流量，即通道0, 3, 4, 7
    time_indices = [0, 3]
    size_indices = [4, 7]

    # --- 执行攻击和评估 ---
    total_samples = 0
    final_time_l2_ratios = []
    final_size_l2_ratios = []
    
    model.eval()
    total_adv_samples = []
    total_original_samples = []
    for (batch_samples,) in tqdm(test_loader, desc="正在攻击测试集"):
        for i in range(batch_samples.size(0)):
            original_sample = batch_samples[i].unsqueeze(0)
            
            # 1. 记录样本数         
            total_samples += 1

            # 2. 执行I-FGSM攻击
            adv_sample = deepcorr_ifgsm_attack(
                model, original_sample, time_indices, size_indices, device,
                epsilon_time=args.epsilon_time,epsilon_size=args.epsilon_size, num_iter=args.num_iter, max_l2_ratio=args.max_l2_ratio
            )

            # 计算最终的L2扰动比率
            with torch.no_grad():
                total_perturbation = adv_sample.cpu() - original_sample
                pert_time = total_perturbation[:, :, time_indices, :]
                pert_size = total_perturbation[:, :, size_indices, :]
                orig_time = original_sample[:, :, time_indices, :]
                orig_size = original_sample[:, :, size_indices, :]

                time_ratio = (torch.linalg.norm(pert_time,ord=2,dim = -1)/ (torch.linalg.norm(orig_time,ord=2,dim = -1))).mean() 
                size_ratio = (torch.linalg.norm(pert_size,ord=2,dim = -1)/ (torch.linalg.norm(orig_size,ord=2,dim = -1))).mean() 
                
                print(f"样本 {total_samples}: 时间特征L2扰动比率: {time_ratio.item():.4f}, 大小特征L2扰动比率: {size_ratio.item():.4f}")
                
                final_time_l2_ratios.append(time_ratio.item())
                final_size_l2_ratios.append(size_ratio.item())
            total_adv_samples.append(adv_sample.cpu())
            total_original_samples.append(original_sample.cpu())   
            
    # --- 打印最终结果 ---
    print("\n--- I-FGSM 攻击评估结果 ---")
    print(f"平均时间特征L2扰动比率: {np.mean(final_time_l2_ratios):.4f}")
    print(f"平均大小特征L2扰动比率: {np.mean(final_size_l2_ratios):.4f}")
    result_fpath = pathlib.Path(f'baseline/I_FGSM/mdeepcorr/IFGSMmdeepcorr_advsamples_time{np.mean(final_time_l2_ratios):.4f}_size{np.mean(final_size_l2_ratios):.4f}.p')
    with open(result_fpath, "wb") as fp:
        results = {
            "total_adv_samples": total_adv_samples,
            "total_original_samples": total_original_samples,
            "time_l2_ratios": np.mean(final_time_l2_ratios),
            "size_l2_ratios": np.mean(final_size_l2_ratios),
        }
        pickle.dump(results, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="I-FGSM attack on DeepCorr")
    parser.add_argument("--model_path", type=str,default = "target_model/deepcorr/deepcorr700/tor700_199_epoch11_acc0.88.pth", help="预训练DeepCorr模型的路径 (.pth文件)")
    parser.add_argument("--data_folder", type=str, default = "target_model/deepcorr/dataset/", help="包含.pickle数据集文件的文件夹路径")
    parser.add_argument("--index_folder", type=str,default = "target_model/mdeepcorr/" , help="包含test_index.pickle文件的文件夹路径")
    parser.add_argument("--device", type=str, default="cuda:1", help="计算设备, e.g., 'cuda:0' or 'cpu'")
    parser.add_argument("--num_samples", type=int, default=1000, help="用于攻击的样本数量")
    
    # --- 攻击超参数 ---
    parser.add_argument("--epsilon_time", type=float, default=2.5, help="I-FGSM的每步扰动大小(步长)")
    parser.add_argument("--epsilon_size", type=float, default=0.025, help="I-FGSM的每步扰动大小(步长)")
    parser.add_argument("--num_iter", type=int, default=200, help="I-FGSM的最大迭代次数")
    parser.add_argument("--max_l2_ratio", type=float, default=0.15, help="L2范数比率的停止阈值")
    
    args = parser.parse_args()
    main(args)