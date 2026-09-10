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
# ------------------------------------------------------------
# 1. 模型定义 (保持不变)
# ------------------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 2000, (2, 20), stride=2)
        self.max_pool1 = nn.MaxPool2d((1, 5), stride=1)
        self.conv2 = nn.Conv2d(2000, 800, (4, 10), stride=2)
        self.max_pool2 = nn.MaxPool2d((1, 3), stride=1)
        self.fc1 = nn.Linear(49600, 3000)
        self.fc2 = nn.Linear(3000, 800)
        self.fc3 = nn.Linear(800, 100)
        self.fc4 = nn.Linear(100, 1)

    def forward(self, inp, dropout=0.0):
        x = inp
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=dropout)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=dropout)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=dropout)
        x = self.fc4(x)
        return x

# ------------------------------------------------------------
# 2. JSMA 攻击函数 (扰动到最大数量版)
# ------------------------------------------------------------
def jsma_attack(model, original_sample, target_label, 
                perturbable_channels,
                time_indices, size_indices, device,
                epsilon_time=5.0, epsilon_size=0.005,
                max_features_to_perturb=100, 
                max_l2_ratio_time=0.1,
                max_l2_ratio_size=0.1,
                verbose=False):
    """
    对单个样本执行JSMA攻击，扰动到最大数量或达到L2限制，不因成功而提前停止。
    """
    model.eval()
    adv_sample = original_sample.clone().detach()
    adv_sample.requires_grad = True

    with torch.no_grad():
        original_time_norm = torch.linalg.norm(original_sample[:, :, time_indices, :])
        original_size_norm = torch.linalg.norm(original_sample[:, :, size_indices, :])
        epsilon = 1e-12
        absolute_max_l2_time = original_time_norm * max_l2_ratio_time + epsilon
        absolute_max_l2_size = original_size_norm * max_l2_ratio_size + epsilon

    score_change_direction = 1.0 if target_label == 1 else -1.0
    perturbed_features_mask = torch.zeros_like(original_sample.squeeze()).bool().to(device)
    search_space_mask = torch.zeros_like(original_sample.squeeze()).bool().to(device)
    search_space_mask[perturbable_channels, :] = True

    for i in range(max_features_to_perturb):
        output = model(adv_sample)
        

        # current_pred = (torch.sigmoid(output).item() > 0.5)
        # if current_pred == bool(target_label):
        #     break
            
        model.zero_grad()
        if adv_sample.grad is not None:
            adv_sample.grad.zero_()
        output.backward()
        grads = adv_sample.grad.data.squeeze().detach()
        
        saliency_map = score_change_direction * grads
        saliency_map[saliency_map < 0] = -float('inf')
        saliency_map[~search_space_mask] = -float('inf')
        saliency_map[perturbed_features_mask] = -float('inf')

        # (这里保留了轮流攻击逻辑，因为它能更好地平衡扰动)
        time_features_mask = torch.zeros_like(saliency_map, dtype=torch.bool)
        time_features_mask[time_indices, :] = True
        size_features_mask = torch.zeros_like(saliency_map, dtype=torch.bool)
        size_features_mask[size_indices, :] = True
        
        attack_time_first = (i % 2 == 0)
        best_saliency_val = torch.tensor(-float('inf')).to(device)
        best_feature_idx_flat = torch.tensor(-1).to(device)

        def find_best_in_category(s_map, category_mask):
            s_map_category = s_map.clone()
            s_map_category[~category_mask] = -float('inf')
            val, idx = s_map_category.view(-1).max(0)
            return val, idx

        if attack_time_first:
            val, idx = find_best_in_category(saliency_map, time_features_mask)
            if not torch.isinf(val):
                best_saliency_val, best_feature_idx_flat = val, idx
            else:
                val, idx = find_best_in_category(saliency_map, size_features_mask)
                best_saliency_val, best_feature_idx_flat = val, idx
        else:
            val, idx = find_best_in_category(saliency_map, size_features_mask)
            if not torch.isinf(val):
                best_saliency_val, best_feature_idx_flat = val, idx
            else:
                val, idx = find_best_in_category(saliency_map, time_features_mask)
                best_saliency_val, best_feature_idx_flat = val, idx
        
        if torch.isinf(best_saliency_val):
            break
            
        channel_idx = torch.div(best_feature_idx_flat, original_sample.shape[3], rounding_mode='floor')
        pos_idx = best_feature_idx_flat % original_sample.shape[3]
        perturbed_features_mask[channel_idx, pos_idx] = True

        if channel_idx in time_indices:
            perturbation_value = epsilon_time
        elif channel_idx in size_indices:
            perturbation_value = epsilon_size
        else:
            perturbation_value = 0.0
        
        perturbation_step = torch.zeros_like(adv_sample)
        perturbation_step[0, 0, channel_idx, pos_idx] = perturbation_value
        
        potential_adv_sample = adv_sample.detach() + perturbation_step
        total_perturbation = potential_adv_sample - original_sample
        
        current_time_l2 = torch.linalg.norm(total_perturbation[:, :, time_indices, :])
        current_size_l2 = torch.linalg.norm(total_perturbation[:, :, size_indices, :])

        if current_time_l2 > absolute_max_l2_time or current_size_l2 > absolute_max_l2_size:
            perturbed_features_mask[channel_idx, pos_idx] = False
            break
        
        adv_sample.data = potential_adv_sample.data
    print(f"攻击完成，扰动特征数量: {perturbed_features_mask.sum().item()}")
    final_perturbation = adv_sample.detach() - original_sample
    # 不再返回 success 标志
    return adv_sample.detach(), final_perturbation

# ------------------------------------------------------------
# 3. 评估主函数
# ------------------------------------------------------------
def evaluate_jsma_on_test_set(model, test_loader, device, attack_params):
    model.eval()
    correctly_classified_originals = 0
    
    total_time_l2_ratio = 0.0
    total_size_l2_ratio = 0.0
    
    time_indices = attack_params['time_indices']
    size_indices = attack_params['size_indices']
    epsilon_l2 = 1e-12
    total_adv_sample = []
    total_final_perturbation = []
    total_original_sample = []
    for (original_batch, labels_batch) in tqdm(test_loader, desc="正在攻击测试集"):
        for i in range(original_batch.size(0)):
            original_sample = original_batch[i].unsqueeze(0).to(device)
            label = labels_batch[i].item()

            with torch.no_grad():
                original_output = model(original_sample)
                original_pred = (torch.sigmoid(original_output).item() > 0)
            
            # 只攻击那些原始分类正确的样本，threshold设为0,也就是全部扰动
            if original_pred == bool(label):
                correctly_classified_originals += 1
                target_label = 1 - int(label)
                
                # 调用修改后的攻击函数
                adv_sample, final_perturbation = jsma_attack(
                    model=model, original_sample=original_sample, target_label=target_label, 
                    device=device, **attack_params
                )
                
                total_adv_sample.append(adv_sample.cpu())
                total_original_sample.append(original_sample.cpu())
                total_final_perturbation.append(final_perturbation.cpu())
                # 累加L2范数比率
                with torch.no_grad():
                    original_time_norm = torch.linalg.norm(original_sample[:, :, time_indices, :])
                    original_size_norm = torch.linalg.norm(original_sample[:, :, size_indices, :])
                    
                    time_pert_l2 = torch.linalg.norm(final_perturbation[:, :, time_indices, :])
                    size_pert_l2 = torch.linalg.norm(final_perturbation[:, :, size_indices, :])
                    
                    total_time_l2_ratio += (time_pert_l2 / (original_time_norm )).item()
                    total_size_l2_ratio += (size_pert_l2 / (original_size_norm )).item()
                    print(f"样本 {i+1}/{original_batch.size(0)}: "
                          f"时间特征 L2 比率: {time_pert_l2 / (original_time_norm ):.4f}, "
                          f"大小特征 L2 比率: {size_pert_l2 / (original_size_norm ):.4f}")

    # 计算平均L2范数
    if correctly_classified_originals > 0:
        avg_time_l2_ratio = total_time_l2_ratio / correctly_classified_originals
        avg_size_l2_ratio = total_size_l2_ratio / correctly_classified_originals
    else:
        avg_time_l2_ratio = avg_size_l2_ratio = 0.0

    print("\n--- 固定数量扰动评估结果 ---")
    print(f"评估样本总数 (原始分类正确): {correctly_classified_originals}")
    print("\n--- 平均L2范数比率 ---")
    print(f"时间特征部分的 L2 范数比率: {avg_time_l2_ratio:.4f}")
    print(f"大小特征部分的 L2 范数比率: {avg_size_l2_ratio:.4f}")
    result_fpath = pathlib.Path(f'baseline/AJSMA/deepcorr/AJSMAdeepcorr_advsamples_time{avg_time_l2_ratio:.4f}_size{avg_size_l2_ratio:.4f}.p')
    with open(result_fpath, "wb") as fp:
        result ={
            "total_adv_sample": total_adv_sample,
            "total_original_sample": total_original_sample,
            "total_final_perturbation": total_final_perturbation,
            "avg_time_l2_ratio": avg_time_l2_ratio,
            "avg_size_l2_ratio": avg_size_l2_ratio,
        }
        pickle.dump(result, fp)
# ------------------------------------------------------------
# 4. 主执行块 (保持不变)
# ------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="JSMA-style attack on DeepCorr (fixed perturbation count)")
    parser.add_argument("--model_path", type=str, default='target_model/deepcorr/deepcorr300/tor_199_epoch23_acc0.82dict.pth', help="预训练DeepCorr模型的路径")
    parser.add_argument("--data_folder", type=str, default='target_model/deepcorr/dataset/', help="包含.pickle数据集文件的文件夹路径")
    parser.add_argument("--index_folder", type=str, default='target_model/deepcorr/deepcorr300/', help="包含test_index300.pickle文件的文件夹路径")
    parser.add_argument("--device", type=str, default='cuda:0', help="计算设备")
    parser.add_argument("--num_samples", type=int, default=1000, help="用于攻击的样本数量")
    
    parser.add_argument("--epsilon_time", type=float, default=60, help="对时间特征的单步固定扰动值")
    parser.add_argument("--epsilon_size", type=float, default=0.5, help="对大小特征的单步固定扰动值")
    parser.add_argument("--max_features_to_perturb", type=int, default=300, help="最大修改特征点数")
    parser.add_argument("--max_l2_ratio_time", type=float, default=0.15, help="时间总扰动的最大L2范数比率")
    parser.add_argument("--max_l2_ratio_size", type=float, default=0.15, help="大小总扰动的最大L2范数比率")

    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    deepcorr_model = Net().to(device)
    try:
        deepcorr_model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"成功从 {args.model_path} 加载预训练模型。")
    except Exception as e:
        print(f"加载模型时出错: {e}")

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
    
    test_index_path = os.path.join(args.index_folder, 'test_index300.pickle')
    test_index = pickle.load(open(test_index_path, 'rb'))[:args.num_samples]

    def generate_positive_samples_only(dataset, test_index, flow_size=300):
        l2s_test = torch.zeros((len(test_index), 1, 8, flow_size)) 
        labels_test = torch.ones(len(test_index))
        for index, i in enumerate(test_index):
            l2s_test[index, 0, 0,:] = torch.tensor(dataset[i]['here'][0]['<-'][:flow_size]) * 1000.0  
            l2s_test[index, 0, 1,:] = torch.tensor(dataset[i]['there'][0]['->'][:flow_size]) * 1000.0  
            l2s_test[index, 0, 2,:] = torch.tensor(dataset[i]['there'][0]['<-'][:flow_size]) * 1000.0  
            l2s_test[index, 0, 3,:] = torch.tensor(dataset[i]['here'][0]['->'][:flow_size]) * 1000.0  

            l2s_test[index, 0, 4,:] = torch.tensor(dataset[i]['here'][1]['<-'][:flow_size]) / 1000.0  
            l2s_test[index, 0, 5,:] = torch.tensor(dataset[i]['there'][1]['->'][:flow_size]) / 1000.0  
            l2s_test[index, 0, 6,:] = torch.tensor(dataset[i]['there'][1]['<-'][:flow_size]) / 1000.0  
            l2s_test[index, 0, 7,:] = torch.tensor(dataset[i]['here'][1]['->'][:flow_size]) / 1000.0  
        return l2s_test, labels_test
    
    test_l2s, test_labels = generate_positive_samples_only(dataset, test_index, flow_size=300)
    test_dataset = TensorDataset(test_l2s.float(), test_labels.float())
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    jsma_parameters = {
        'epsilon_time': args.epsilon_time,
        'epsilon_size': args.epsilon_size,
        'max_features_to_perturb': args.max_features_to_perturb,   
        'max_l2_ratio_time': args.max_l2_ratio_time,       
        'max_l2_ratio_size': args.max_l2_ratio_size,      
        'perturbable_channels': [0, 3, 4, 7], 
        'time_indices': [0, 3], 
        'size_indices': [4, 7]  
    }
    
    evaluate_jsma_on_test_set(deepcorr_model, test_loader, device, jsma_parameters)