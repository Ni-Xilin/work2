'''
Random Mutation:generates evasive packet sequences by randomly varying inter-packet interval time within a packet
sequence iteratively. This method is not a baseless weak attack but is recognized in prior works.

'''
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
    
flow_size = 300
# l2s_test = torch.zeros(len(dataset),1, 8, flow_size)
# for index in range(len(dataset)):
#     l2s_test[index, 0, 0,:] = torch.tensor(dataset[index]['here'][0]['<-'][:flow_size]) * 1000.0  
#     l2s_test[index, 0, 1,:] = torch.tensor(dataset[index]['there'][0]['->'][:flow_size]) * 1000.0  
#     l2s_test[index, 0, 2,:] = torch.tensor(dataset[index]['there'][0]['<-'][:flow_size]) * 1000.0  
#     l2s_test[index, 0, 3,:] = torch.tensor(dataset[index]['here'][0]['->'][:flow_size]) * 1000.0  

#     l2s_test[index, 0, 4,:] = torch.tensor(dataset[index]['here'][1]['<-'][:flow_size]) / 1000.0  
#     l2s_test[index, 0, 5,:] = torch.tensor(dataset[index]['there'][1]['->'][:flow_size]) / 1000.0  
#     l2s_test[index, 0, 6,:] = torch.tensor(dataset[index]['there'][1]['<-'][:flow_size]) / 1000.0  
#     l2s_test[index, 0, 7,:] = torch.tensor(dataset[index]['here'][1]['->'][:flow_size]) / 1000.0  

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
    # print('L2 rate,time:%.3f,size:%.3f' % (time_rate, size_rate))
    return adv_data,adv,time_rate.item(),size_rate.item()

negetive_samples = 199
#使用随机扰动的方式来生成数据，取扰动大小为(0,data[0][0][0][:flow_size].mean()),为了保证一致性，也只扰动客户端的时间数据
def generate_data_true(dataset, test_index, flow_size):
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
        
def generate_data(total_adv_samples, flow_size):
    negetive_samples = 199
    dataset = torch.cat(total_adv_samples, dim=0)  # Assuming total_adv_samples is a list of tensors
    test_index = list(range(len(dataset)))
    #测试集样本填充(all_samples*(negetive_samples+1),1,8,flow_size)
    index = 0
    random_test = [] + test_index
    l2s_test = torch.zeros((len(dataset) * (negetive_samples + 1),1, 8, flow_size)) 
    labels_test = torch.zeros((len(dataset) * (negetive_samples + 1))) 

    for i in test_index:
        if index % (negetive_samples + 1) != 0:
            print(index, len(random_test))
            raise ValueError("Index is not a multiple of (negetive_samples + 1)")
        m = 0

        np.random.shuffle(random_test)
        for idx in random_test:
            if idx == i or m > (negetive_samples - 1):
                continue

            m += 1
            l2s_test[index, 0, 0,:] = dataset[idx,0,0,:flow_size] 
            l2s_test[index, 0, 1,:] = dataset[i,0,1,:flow_size]
            l2s_test[index, 0, 2,:] = dataset[i,0,2,:flow_size] 
            l2s_test[index, 0, 3,:] = dataset[idx,0,3,:flow_size] 

            l2s_test[index, 0, 4,:] = dataset[idx,0,4,:flow_size]   
            l2s_test[index, 0, 5,:] = dataset[i,0,5,:flow_size] 
            l2s_test[index, 0, 6,:] = dataset[i,0,6,:flow_size]  
            l2s_test[index, 0, 7,:] = dataset[idx,0,7,:flow_size] 
            labels_test[index] = 0
            index += 1

        l2s_test[index, 0, 0,:] = dataset[i,0,0,:flow_size]  
        l2s_test[index, 0, 1,:] = dataset[i,0,1,:flow_size] 
        l2s_test[index, 0, 2,:] = dataset[i,0,2,:flow_size] 
        l2s_test[index, 0, 3,:] = dataset[i,0,3,:flow_size] 

        l2s_test[index, 0, 4,:] = dataset[i,0,4,:flow_size] 
        l2s_test[index, 0, 5,:] = dataset[i,0,5,:flow_size] 
        l2s_test[index, 0, 6,:] = dataset[i,0,6,:flow_size]  
        l2s_test[index, 0, 7,:] = dataset[i,0,7,:flow_size]  
        labels_test[index] = 1

        index += 1
    return l2s_test, labels_test

flow_size = 300

batch_size = 256
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 2000, (2,20), stride=2)
        self.max_pool1 = nn.MaxPool2d((1,5), stride=1)
        self.conv2 = nn.Conv2d(2000, 800, (4,10), stride=2)
        self.max_pool2 = nn.MaxPool2d((1,3), stride=1)
        self.fc1 = nn.Linear(49600, 3000)
        self.fc2 = nn.Linear(3000, 800)
        self.fc3 = nn.Linear(800, 100)
        self.fc4 = nn.Linear(100, 1)
#         self.d = nn.Dropout2d()
    
    def weight_init(self):
        for m in self._modules:
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.zero_()
                
    def forward(self, inp, dropout):
        x = inp
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=dropout)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=dropout)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=dropout)
        x = self.fc4(x)
        return x
    
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
print (device)


model = Net().to(device)
LOADED=torch.load('target_model/deepcorr/deepcorr300/tor_199_epoch23_acc0.82dict.pth',map_location=device)
model.load_state_dict(LOADED)

test_index = pickle.load(open('target_model/deepcorr/deepcorr300/test_index300.pickle','rb'))[:1000]

# 1. 收集所有模型的预测概率和真实标签
all_outputs = []
all_labels = []
result_fpath = pathlib.Path("baseline/RM/deepcorr/RMtest_index300_result.p")

row = [0,3]  
# if result_fpath.exists():
if False:
    with open(result_fpath, "rb") as fp:
        all_outputs,all_labels = pickle.load(fp)
else:
    
    test_l2s, test_labels = generate_data_true(dataset, test_index,  flow_size)
    test_dataset = TensorDataset(test_l2s.float(), test_labels.float())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
    
    total_adv_samples = []
    total_time_ratio = []
    total_szie_ratio = []
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        RMdata,adv,time_ratio,size_ratio = RM(data,device)
        total_adv_samples.append(RMdata)
        total_time_ratio.append(time_ratio)
        total_szie_ratio.append(size_ratio)
    time_ratio = torch.tensor(total_time_ratio).mean().item()
    size_ratio = torch.tensor(total_szie_ratio).mean().item()   
    test_l2s, test_labels = generate_data(total_adv_samples, flow_size)
    test_dataset = TensorDataset(test_l2s.float(), test_labels.float())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
    
    model.eval() 
    with torch.no_grad():
        for data, labels in tqdm(test_loader):
            data, labels = data.to(device), labels.to(device)
            
            #data: [batch_size, 1, 8, flow_size]
            # 得到模型经过RM扰动的输出
            # RMdata,adv = RM(data,device)
            # print('L2 distance of the RM dataset: %.3f' % torch.linalg.norm( data[:,0,row,:], ord=2,dim=(-1)).mean())
            # print('L2 distance of the adv: %.3f' % torch.linalg.norm(adv, ord=2,dim=(-1)).mean())
            # print('L2 rate %.3f' % (torch.linalg.norm(adv, ord=2,dim=(-1)).mean() / torch.linalg.norm(data[:,0,row,:], ord=2,dim=(-1)).mean()))
            
            outputs_raw = model(data, dropout=0.0)

            # 使用 sigmoid 转换为概率
            outputs_prob = torch.sigmoid(outputs_raw).cpu().numpy()
            
            # 收集概率和标签
            all_outputs.extend(outputs_prob.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # 转换为numpy数组
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    print(f'L2 ratio of the RM datase:time:{time_ratio},size{size_ratio}')
    with open(result_fpath, "wb") as fp:
        pickle.dump((all_outputs,all_labels,time_ratio,size_ratio), fp)