from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import warnings
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
# from torchvision import datasets, transforms
import tqdm
import pickle
import pathlib
import random
# --- 用于可复现性的种子(SEED)设置 ---
SEED = 42  # 你可以将这个数字更改为任何其他整数

random.seed(SEED)  # 为 Python 内置的 random 模块设置种子
np.random.seed(SEED)  # 为 NumPy 库设置种子
torch.manual_seed(SEED)  # 为 PyTorch 的 CPU 操作设置种子

# 如果你的电脑支持并正在使用 NVIDIA GPU (CUDA)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)  # 为所有 GPU 设置种子
    # 为了在 CUDA 上实现完全的可复现性，下面这两行代码可能是必需的，
    # 但它们可能会对训练速度产生负面影响。
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
# ---------------------------------------------------------

class PacketWithSizeFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, noise,sizes, inp  , num):
        num = int(num)
        if num ==0:
            return inp
        
        tops = torch.argsort(noise,descending=False)
        
        perts = generate_perturbation(tops[:num])
        
        
        output = inp.clone() # 使用clone避免in-place修改问题
        
        # --- MODIFICATION START ---
        # 决定新包的特征值
        time_adv_value = torch.ones_like(noise[tops[:num]])*0.001
        size_adv_value = 0.595*((sizes[tops[:num]]>0).float()+1)

        # 移动所有通道的数据为新包腾出空间
        # 这里假设所有8个通道都需要平移以保持对齐
        for ch in range(8):
             output[:,:,ch,:] = output[:,:,ch,perts]
        
        # 在Tor端的所有相关通道上注入新包的特征
        # 时间通道: 0 和 3
        output[:,:,0,tops[:num]] = time_adv_value
        output[:,:,3,tops[:num]] = time_adv_value # 新增

        # 大小通道: 4 和 7
        output[:,:,4,tops[:num]] = size_adv_value
        output[:,:,7,tops[:num]] = size_adv_value # 新增
        # --- MODIFICATION END ---
    
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # ... backward部分保持不变 ...
        return grad_output[:,0,0,:].sum(dim=0),grad_output[:,0,4,:].sum(dim=0), grad_output , None


class SizePaddingRemapping(torch.autograd.Function):
    """
    Implements the size remapping function from Algorithm 3 in the paper.
    This function is non-differentiable, so we use a custom backward pass.
    """
    @staticmethod
    def forward(ctx, raw_perturbation, data_in, total_overhead_bytes, per_packet_overhead_bytes):
        
        # We don't want to modify the original data tensor in-place
        data_out = data_in.clone()
        
        if total_overhead_bytes == 0:
            return data_out

        # As per Algorithm 3, we select packets with the highest adversarial values to perturb.
        # The algorithm sorts in descending order, so we do the same.
        indices_to_perturb = torch.argsort(raw_perturbation, descending=True)
        
        remaining_overhead = total_overhead_bytes
        
        for i in indices_to_perturb:
            if remaining_overhead <= 0:
                break
            
            # Calculate the amount of padding to add to this packet
            # Following min() logic in Algorithm 3
            delta = min(per_packet_overhead_bytes, remaining_overhead)
            
            # Update the total remaining overhead
            remaining_overhead -= delta
            
            # MODIFICATION: Add padding only to Tor-side size channels (4 and 7)
            data_out[:, 0, 4, i] += delta / 1000.0 # Normalization factor from data generation
            data_out[:, 0, 7, i] += delta / 1000.0

        ctx.save_for_backward(raw_perturbation)
        return data_out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Custom gradient implementation.
        MODIFICATION: Pass back gradients only from Tor-side size channels (4 and 7).
        """
        grad_pert = grad_output[:, 0, 4, :] + grad_output[:, 0, 7, :]
        return grad_pert, None, None, None


size_pad_remapper = SizePaddingRemapping.apply
    

def generate_data(dataset,train_index,test_index,flow_size):
    


    global negetive_samples



    all_samples=len(train_index)
    labels=np.zeros((all_samples*(negetive_samples+1),1))
    l2s=np.zeros((all_samples*(negetive_samples+1),1,8,flow_size))

    index=0
    random_ordering=[]+train_index
    for i in tqdm.tqdm( train_index):
        #[]#list(lsh.find_k_nearest_neighbors((Y_train[i]/ np.linalg.norm(Y_train[i])).astype(np.float64),(50)))

        l2s[index,0,0,:]=np.array(dataset[i]['here'][0]['<-'][:flow_size])*1000.0
        l2s[index,0,1,:]=np.array(dataset[i]['there'][0]['->'][:flow_size])*1000.0
        l2s[index,0,2,:]=np.array(dataset[i]['there'][0]['<-'][:flow_size])*1000.0
        l2s[index,0,3,:]=np.array(dataset[i]['here'][0]['->'][:flow_size])*1000.0

        l2s[index,0,4,:]=np.array(dataset[i]['here'][1]['<-'][:flow_size])/1000.0
        l2s[index,0,5,:]=np.array(dataset[i]['there'][1]['->'][:flow_size])/1000.0
        l2s[index,0,6,:]=np.array(dataset[i]['there'][1]['<-'][:flow_size])/1000.0
        l2s[index,0,7,:]=np.array(dataset[i]['here'][1]['->'][:flow_size])/1000.0


        if index % (negetive_samples+1) !=0:
            print (index , len(nears))
            raise
        labels[index,0]=1
        m=0
        index+=1
        np.random.shuffle(random_ordering)
        for idx in random_ordering:
            if idx==i or m>(negetive_samples-1):
                continue

            m+=1

            l2s[index,0,0,:]=np.array(dataset[idx]['here'][0]['<-'][:flow_size])*1000.0
            l2s[index,0,1,:]=np.array(dataset[i]['there'][0]['->'][:flow_size])*1000.0
            l2s[index,0,2,:]=np.array(dataset[i]['there'][0]['<-'][:flow_size])*1000.0
            l2s[index,0,3,:]=np.array(dataset[idx]['here'][0]['->'][:flow_size])*1000.0

            l2s[index,0,4,:]=np.array(dataset[idx]['here'][1]['<-'][:flow_size])/1000.0
            l2s[index,0,5,:]=np.array(dataset[i]['there'][1]['->'][:flow_size])/1000.0
            l2s[index,0,6,:]=np.array(dataset[i]['there'][1]['<-'][:flow_size])/1000.0
            l2s[index,0,7,:]=np.array(dataset[idx]['here'][1]['->'][:flow_size])/1000.0

            #l2s[index,0,:,0]=Y_train[i]#np.concatenate((Y_train[i],X_train[idx]))#(Y_train[i]*X_train[idx])/(np.linalg.norm(Y_train[i])*np.linalg.norm(X_train[idx]))
            #l2s[index,1,:,0]=X_train[idx]



            labels[index,0]=0
            index+=1
    return l2s, labels

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
    
    def forward(self, inp, dropout_prob):
        x = inp
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        
        x = x.view(batch_size, -1)
        
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=dropout_prob)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=dropout_prob)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=dropout_prob)
        x = self.fc4(x)
        return x
        
        
class TIMENOISER(nn.Module):
    def __init__(self,inp):
        super(TIMENOISER, self).__init__()

        self.inp = inp

        self.independent = nn.Sequential(
            nn.Linear(inp,500),
            nn.ReLU(),
            nn.Linear(500,inp)
        )
#         self.d = nn.Dropout2d()
    
    def weight_init(self):
        for m in self._modules:
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, inp,eps,mid,outsize = 300):
        x = inp[:,:self.inp]
        nz = torch.ones_like(x)
        nz.uniform_(-2,2)
        ind = self.independent(nz)
    
        res = ind
        res = transfer_adv(res,eps,mid)
        
        res = res.view(-1,1,self.inp)
        
        
        
        if self.inp < outsize :
            z = torch.zeros_like(res)
            res = torch.cat([res,z,z,z,z,z],dim=2)
            res = res[:,:,:outsize]
            
        
        z = torch.zeros_like(res)
        # MODIFICATION: Apply time perturbation 'res' to channels 0 and 3.
        # Channels:         0, 1, 2, 3,   4, 5, 6, 7
        x = torch.stack([res, z, z, res,  z, z, z, z], dim=2)
        
        
        return x

class SIZEPADNOISER(nn.Module):
    """
    Generates raw perturbation vector for size padding, as per Table 1 in the paper.
    """
    def __init__(self, inp):
        super(SIZEPADNOISER, self).__init__()
        self.inp = inp
        self.z = torch.FloatTensor(size=(1, inp))
        self.z = self.z.to(device)
        self.generator = nn.Sequential(
            nn.Linear(inp, 500),
            nn.ReLU(),
            nn.Linear(500, inp)
        )

    def forward(self, outsize=300):
        nz = self.z
        nz.uniform_(-0, 0.5)
        raw_pert = self.generator(nz)
        
        if self.inp < outsize:
            z = torch.zeros_like(raw_pert)
            raw_pert = torch.cat([raw_pert, z, z, z, z, z], dim=1)
        
        return raw_pert[:, :outsize].view(-1)


class ADDNOISER(nn.Module):
    def __init__(self,inp,device):
        super(ADDNOISER, self).__init__()
        
        self.inp = inp
        
        self.z = torch.FloatTensor(size=(1,inp))
        self.z =self.z.to(device)
        self.independent_where = nn.Sequential(
            nn.Linear(inp,500),
            nn.ReLU(),
            nn.Linear(500,inp)
        )
        self.independent_size = nn.Sequential(
            nn.Linear(inp,500),
            nn.ReLU(),
            nn.Linear(500,inp)
        )
#         self.d = nn.Dropout2d()
    
    def weight_init(self):
        for m in self._modules:
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.zero_()
    
    def forward(self,outsize=300):
        
        nz = self.z
        nz.uniform_(-0,0.5)
        #print (nz.shape)
        ind_where = self.independent_where(nz)
        ind_size = self.independent_size(nz)
        
        if self.inp < outsize :
            z = torch.zeros_like(ind_where)
            ind_size = torch.cat([ind_size,z,z,z,z,z],dim=1)
            ind_where = torch.cat([ind_where,z,z,z,z,z],dim=1)
        ind_size = ind_size[:,:outsize]
        ind_where = ind_where[:,:outsize]
        
        
        
        return ind_where.view(-1),ind_size.view(-1)
        
        
class discrim(nn.Module):
    def __init__(self,inp):
        super(discrim, self).__init__()
        self.inp = inp
        self.dependent = nn.Sequential(
                nn.Linear(inp,1000),
                nn.ReLU(),
                nn.Linear(1000,1000),
                nn.ReLU(),
                nn.Linear(1000,1)
            )
    
    
    def forward(self, inp):
        return  self.dependent(inp[:,0,0,:self.inp])

def generate_perturbation(change_points, size = 300):
    start = size-len(change_points)
    pert = [] 
    passed = 0 
    for ind in range(size):
        if ind in change_points:
            pert.append(start)
            start +=1
            passed+=1
        else:
            pert.append(ind-passed)
        
    return pert



decider= PacketWithSizeFunction.apply
def train_adv(adv_model, size_add_model, size_pad_model, optim_adv, disc_model, opt_dis, model, device, data, label,
              num_to_add=0, reg=0, eps=0.3, mid=5.0, deps=300,
              size_padding_total_kb=0, size_padding_packet_bytes=0):
    model.eval()
    model.zero_grad()
    
    # --- Perturbation Step 1: Add new packets ---
    where, sizes = size_add_model()
    data_adv = decider(where, sizes, data, num_to_add)

    # --- Perturbation Step 2: Pad existing packet sizes ---
    raw_size_pad_pert = size_pad_model()
    total_overhead_bytes = size_padding_total_kb * 1024
    data_adv_padded = size_pad_remapper(raw_size_pad_pert, data_adv, total_overhead_bytes, size_padding_packet_bytes)
    
    # If using timing regularization, perform GAN training step
    if reg > 0:
        adv = adv_model(data_adv_padded[:,0,0,:deps], eps, mid)
        z = torch.zeros_like(adv)
        z.normal_(mean=mid, std=eps)
        fake = disc_model(adv)
        real = disc_model(z)
        
        f_loss = F.binary_cross_entropy_with_logits(fake, torch.zeros_like(label))
        r_loss = F.binary_cross_entropy_with_logits(real, torch.ones_like(label))

        d_loss = f_loss + r_loss
        opt_dis.zero_grad()
        d_loss.backward(retain_graph=True) # Retain graph for the main loss backward pass
        opt_dis.step()

    # --- Perturbation Step 3: Perturb Timings ---
    adv_time = adv_model(data_adv_padded[:,0,0,:deps], eps, mid)
    final_perturbed_data = torch.clamp(data_adv_padded + adv_time, min=0)
    
    output = model(final_perturbed_data, 0.0)
    
    # Calculate main adversarial loss
    loss = F.binary_cross_entropy_with_logits(output, 1 - label)
    
    # Add GAN regularizer loss for timing
    if reg > 0:
        fake_for_gen = disc_model(adv_time)
        loss += reg * F.binary_cross_entropy_with_logits(fake_for_gen, torch.ones_like(label))
    optim_adv.zero_grad()
    loss.backward()
    optim_adv.step()
    return loss.item()
    
def gen_advs(adv_model,optim_adv,disc_model,opt_dis, model, device, data, label, lr, alpha=0.5, eps=0.3,mid=5.0,deps = 300):
    model.eval()
    model.zero_grad()
    
    #data[:,:,0,:] = torch.clamp(data[:,:,0,:] + transfer_adv(adv_model,eps,mid), min = 0.0)
    adv  = adv_model(data[:,0,0,:deps],eps,mid)
    z = torch.zeros_like(adv )
    z.normal_(mean=mid,std=eps)
    return adv,z
    
    

    
#     print ('Mean:  ',transfer_adv(adv_model, eps, mid).mean())
#     print ('STD:   ',transfer_adv(adv_model, eps, mid).std())


def test_adv(adv_model, size_add_model, size_pad_model, model, device, data, num_to_add, eps, mid, deps,
             size_padding_total_kb=0, size_padding_packet_bytes=0):
    model.eval()
    model.zero_grad()

    # --- Perturbation Step 1: Add new packets ---
    where, sizes = size_add_model()
    data_adv = decider(where, sizes, data, num_to_add)
    
    # --- Perturbation Step 2: Pad existing packet sizes ---
    raw_size_pad_pert = size_pad_model()
    total_overhead_bytes = size_padding_total_kb * 1024
    data_adv_padded = size_pad_remapper(raw_size_pad_pert, data_adv, total_overhead_bytes, size_padding_packet_bytes)

    # --- Perturbation Step 3: Perturb Timings ---
    adv_time = adv_model(data_adv_padded[:,0,0,:deps], eps, mid)
    final_perturbed_data = torch.clamp(data_adv_padded + adv_time, min=0)
    
    output_adv = model(final_perturbed_data, 0.0)
    
    o = torch.sigmoid(output_adv)
    return o
    
    
def total_test(adv_model, size_add_model, size_pad_model, model, device, num_to_add, eps, mid, deps,
               size_padding_total_kb=0, size_padding_packet_bytes=0):
    global test_index
    global dataset
    global batch_size
    a = -1
    corrs=np.zeros((500,500))
    batch=[]
    l2s_test_all=np.zeros((batch_size,1,8,flow_size))
    l_ids=[]
    index=0
    xi,xj=0,0
    for i in (test_index[:500]):
        xj=0
        for j in test_index[:500]:

            l2s_test_all[index,0,0,:]=np.array(dataset[j]['here'][0]['<-'][:flow_size])*1000.0
            l2s_test_all[index,0,1,:]=np.array(dataset[i]['there'][0]['->'][:flow_size])*1000.0
            l2s_test_all[index,0,2,:]=np.array(dataset[i]['there'][0]['<-'][:flow_size])*1000.0
            l2s_test_all[index,0,3,:]=np.array(dataset[j]['here'][0]['->'][:flow_size])*1000.0

            l2s_test_all[index,0,4,:]=np.array(dataset[j]['here'][1]['<-'][:flow_size])/1000.0
            l2s_test_all[index,0,5,:]=np.array(dataset[i]['there'][1]['->'][:flow_size])/1000.0
            l2s_test_all[index,0,6,:]=np.array(dataset[i]['there'][1]['<-'][:flow_size])/1000.0
            l2s_test_all[index,0,7,:]=np.array(dataset[j]['here'][1]['->'][:flow_size])/1000.0
            l_ids.append((xi,xj))
            index+=1
            if index==batch_size:
                index=0
                test_data = torch.from_numpy(l2s_test_all).float().to(device)
                cor_vals=test_adv(adv_model, size_add_model, size_pad_model, model, device, test_data, num_to_add, eps, mid, deps,
                                  size_padding_total_kb, size_padding_packet_bytes)
                cor_vals = cor_vals.data.cpu().numpy()
                for ids in range(len(l_ids)):
                    di,dj=l_ids[ids]
                    corrs[di,dj]=cor_vals[ids].item()
                l_ids=[]
            xj+=1
        xi+=1
    return (corrs)

negetive_samples=199
flow_size=300
TRAINING= True


all_runs={'8872':'192.168.122.117',
           '8802':'192.168.122.117','8873':'192.168.122.67','8803':'192.168.122.67',
          '8874':'192.168.122.113','8804':'192.168.122.113','8875':'192.168.122.120',
         '8876':'192.168.122.30','8877':'192.168.122.208','8878':'192.168.122.58'}


dataset=[]

for name in all_runs:
    dataset+=pickle.load(open('target_model/deepcorr/dataset/%s_tordata300.pickle'%name, 'rb'))
    
if TRAINING:
    
    
    len_tr=len(dataset)
    train_ratio=float(len_tr-3000)/float(len_tr)
    rr= list(range(len(dataset)))
    np.random.shuffle(rr)

    train_index=rr[:int(len_tr*train_ratio)]
    test_index= rr[int(len_tr*train_ratio):int(len_tr*train_ratio)+500] #range(len(dataset_test)) # #
    # pickle.dump(test_index,open('test_index300.pickle','wb'))
    pickle.dump(train_index, open('baseline/BLANKET/model_blanketdeepcorr/train_index300.pickle', 'wb'))
else:
    test_index=pickle.load(open('test_index300.pickle'))
    
    
parser = argparse.ArgumentParser(description='DEEPCORR BLIND ADV EXAMPLE')

# parser.add_argument('--gpu-id', type=int, default=0, help='Train model')
parser = argparse.ArgumentParser(description='为DeepCorr模型训练盲对抗性扰动生成器')

# --- 训练与模型结构设置 ---
parser.add_argument('--input-size', type=int, default=300,
                    help='输入流量特征向量的长度 (代码中针对DeepCorr设为300)。')
parser.add_argument('--epochs', type=int, default=20,
                    help='扰动生成模型的总训练周期数。')

# --- 时间扰动参数 (Timing Perturbation) ---
parser.add_argument('--mid', type=float, default=15.0,
                    help="对抗性时间延迟的均值(ms), 对应论文中的'μ'。设置为0通常用于产生隐蔽性更好的抖动攻击。")
parser.add_argument('--sigma', type=float, default=50.0,
                    help="对抗性时间延迟的标准差(ms), 对应论文中的'σ'。根据图4, σ=50ms时对DeepCorr攻击效果最强。")
parser.add_argument('--justpos', type=int, default=1,
                    help='设置为1则只产生正延迟(推迟), 设置为0则允许延迟和提前，攻击能力更强。')

# --- 注入与填充参数 (Injection & SizePadding) ---
parser.add_argument('--to-add', type=float, default=50.0,
                    help='注入到流量中的新虚拟数据包数量。根据图7, 对DeepCorr注入50个包在混合攻击中效果最显著。')
parser.add_argument('--size-padding-total-kb', type=float, default=100.0,
                    help="通过填充现有数据包增加的总带宽开销(KB), 对应论文中的'N'。根据图6, N=100KB时对DeepCorr攻击效果最强。")
parser.add_argument('--size-padding-packet-bytes', type=float, default=512.0,
                    help="允许为单个数据包添加的最大填充字节数'n'。")

# --- 优化器与正则化参数 ---
parser.add_argument('--gen-lr', type=float, default=0.001,
                    help='扰动生成器模型的学习率 (根据论文表1)。')
parser.add_argument('--dis-lr', type=float, default=0.0001,
                    help='判别器模型的学习率, 用于时间分布的正则化 (根据论文表1)。')
parser.add_argument('--similarity-reg', type=float, default=0.0,
                    help='时间分布正则化的系数。设为0代表不使用隐身约束，此时攻击效果最强。')

    
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu", 3)
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
print (device)


model = Net().to(device)
LOADED=torch.load('target_model/deepcorr/deepcorr300/tor_199_epoch23_acc0.82dict.pth', map_location=device)
model.load_state_dict(LOADED)


num_epoches = 200
batch_size = 100


l2s,labels=generate_data(dataset=dataset,train_index=train_index,test_index=test_index,flow_size=flow_size)
rr = list(range(len(l2s)))
np.random.shuffle(rr)
l2s = l2s[rr]
labels = labels[rr]

num_steps = (len(l2s)//batch_size)


###### PARAMETERS
args = parser.parse_args()

inpsize = args.input_size
noise_lr = args.gen_lr
disc_lr = args.dis_lr
# gpu_id = args.gpu_id
mid = args.mid
sigma = args.sigma
num_to_add  = args.to_add
size_padding_total_kb = args.size_padding_total_kb
size_padding_packet_bytes = args.size_padding_packet_bytes
reg = args.similarity_reg
epochs = args.epochs


def transfer_adv(inp,eps,mid):
#     x = 100*F.sigmoid(inp)
#     return F.relu(inp)
    
    x = inp
    if args.justpos == 1 :
        x = F.relu(x)
    res = ((x-torch.clamp(x.mean(dim=1,keepdim=True)-mid,min=0)-torch.clamp(x.mean(dim=1,keepdim=True)+mid,max=0)))

    res_multi = (torch.clamp(x.std(dim=1,keepdim=True),max=eps)/(x.std(dim=1,keepdim=True))+0.00000001)
    res = res * res_multi
    #print (res)
    return  res



SAVE_PATH='baseline/BLANKET/model_blanketdeepcorr/mid%d_sigma%d_numadd%d_alladdsize%.2f_reg%.2f_inpsize%d_just_positive%d/'%(mid,sigma,num_to_add,size_padding_total_kb,reg,inpsize,args.justpos)


pathlib.Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)


timenois = TIMENOISER(inpsize)
timenois.to(device)

addnois = ADDNOISER(inpsize,device)
addnois.to(device)

sizepadnois = SIZEPADNOISER(inpsize)
sizepadnois.to(device)

optim_nos = optim.Adam(list(addnois.parameters())+list(timenois.parameters())+list(sizepadnois.parameters()),lr=noise_lr)

disc  = discrim(inpsize)
disc.to(device)
optim_dis = optim.Adam(disc.parameters(),lr=disc_lr)
print (num_steps)


total_corrs = []
avg_loss = 0.0
for epoch in range(epochs):
    rr = list(range(len(l2s)))
    np.random.shuffle(rr)
    l2s = l2s[rr]
    labels = labels[rr]
    print ('EPOCH %d'%epoch)
    
    loss_window = [] 
    flag = False
    for step in range(500):
        start_ind = step*batch_size
        end_ind = ((step + 1) *batch_size)
        if end_ind < start_ind:
            print ('HOOY')
            continue
            
        else:
            batch_flow = torch.from_numpy(l2s[start_ind:end_ind, :]).float().to(device)
            batch_label = torch.from_numpy(labels[start_ind:end_ind]).float().to(device)
    
        loss =train_adv(timenois, addnois, sizepadnois, optim_nos, disc, optim_dis,
                  model, device, batch_flow, batch_label,
                  num_to_add, reg, sigma, mid, 300,
                  size_padding_total_kb, size_padding_packet_bytes)
        # 记录当前step的loss（假设loss是单个标量值）
        loss_window.append(loss)

        # 每100个step计算一次均值
        if (step + 1) % 100 == 0:
            avg_loss = sum(loss_window) / len(loss_window)
            if(avg_loss < 12):
                # flag = True
                torch.save({'time_model':timenois.state_dict(), 'add_model':addnois.state_dict(), 'sizepad_model':sizepadnois.state_dict()}, SAVE_PATH+'/model_epoch%d_avgloss%.2f'%(epoch,avg_loss))
                # total_corrs.append(corrs)
                print("saving..")
            print(f"Steps [{step-99}-{step}] Average Loss: {avg_loss:.4f}")
            loss_window.clear() 
    # corrs = total_test(timenois, addnois, sizepadnois, model, device,
    #                 num_to_add, sigma, mid, 300,
    #                 size_padding_total_kb, size_padding_packet_bytes)
        
    torch.save({'time_model':timenois.state_dict(), 'add_model':addnois.state_dict(), 'sizepad_model':sizepadnois.state_dict()}, SAVE_PATH+'/model_epoch%d_avgloss%.2f'%(epoch,avg_loss))
    # total_corrs.append(corrs)
    print("saving..")
    # torch.save({'corrs':total_corrs},SAVE_PATH+'/cors')
