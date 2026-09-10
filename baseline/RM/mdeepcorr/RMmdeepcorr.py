import numpy as np
from tqdm import tqdm
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
import pathlib
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

dataset = []

all_runs={'8872':'192.168.122.117','8802':'192.168.122.117','8873':'192.168.122.67','8803':'192.168.122.67',
         '8874':'192.168.122.113','8804':'192.168.122.113','8875':'192.168.122.120',
        '8876':'192.168.122.30','8877':'192.168.122.208','8878':'192.168.122.58'}

for name in all_runs:
    dataset += pickle.load(open('target_model/deepcorr/dataset/%s_tordata300.pickle' % name, 'rb'))
    
flow_size = 700

#L2 rate 约为0.137
def RM(data,device):
    row = [0,3,4,7]
    row_time = [0,3]
    row_size = [4,7]
    adv = torch.rand_like(data[:, 0, row, :]).to(device)
    adv= torch.mul(adv, data[:, 0, row, :])*0.23
    # adv=torch.clamp(adv,0*data[:,0,row,:],1*data[:,0,row,:])
    adv_data = data.clone()
    adv_data[:, 0, row, :] = adv_data[:, 0, row, :] + adv
    time_rate = (torch.linalg.norm(adv_data[:,0,row_time,:]-data[:,0,row_time,:], ord=2,dim=(-1)) / torch.linalg.norm(data[:,0,row_time,:], ord=2,dim=(-1))).mean()
    size_rate = (torch.linalg.norm(adv_data[:,0,row_size,:]-data[:,0,row_size,:], ord=2,dim=(-1)) / torch.linalg.norm(data[:,0,row_size,:], ord=2,dim=(-1))).mean()
    return adv_data,adv,time_rate.item(),size_rate.item()

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
        
    return l2s_test, labels_test

batch_size =64
    
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
print (device)


test_index = pickle.load(open('target_model/mdeepcorr/test_index.pickle','rb'))[:1000]


    
test_l2s, test_labels = generate_positive_samples_only(dataset, test_index,  flow_size)
test_dataset = TensorDataset(test_l2s.float(), test_labels.float())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)

total_original_samples = []
total_final_perturbation = []
total_adv_samples = []
total_time_ratio = []
total_szie_ratio = []
for data, labels in tqdm(test_loader):
    data, labels = data.to(device), labels.to(device)
    RMdata,adv,time_ratio,size_ratio = RM(data,device)
    total_original_samples.append(data)
    total_adv_samples.append(RMdata)
    total_final_perturbation.append(adv)
    total_time_ratio.append(time_ratio)
    total_szie_ratio.append(size_ratio)
time_ratio = torch.tensor(total_time_ratio).mean().item()
size_ratio = torch.tensor(total_szie_ratio).mean().item()   

print(f'L2 ratio of the RM datase:time:{time_ratio},size{size_ratio}')

result_fpath = pathlib.Path(f'baseline/RM/mdeepcorr/RMmdeepcorr_advsamples_time{time_ratio:.4f}_size{size_ratio:.4f}.p')
with open(result_fpath, "wb") as fp:
    result ={
        "adv_samples": total_adv_samples,
        "original_samples": total_original_samples,
        "final_perturbation": total_final_perturbation,
        "avg_time_l2_ratio": time_ratio,
        "avg_size_l2_ratio": size_ratio,
    }
    pickle.dump(result, fp)