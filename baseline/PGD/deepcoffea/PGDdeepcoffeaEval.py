import argparse
from data_provider.data_factory import data_provider
from exp.exp_basic_deepcoffea import Exp_Basic
from target_model import Deepcoffea
from utils.metrics import metric
import numpy as np
import torch.nn.functional as F
import torch
from tqdm import tqdm
import pickle
import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_provider.data_loader import DeepCoffeaDataset
from sklearn.metrics.pairwise import cosine_similarity

import os
import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

def main(delta, win_size, n_wins, threshold, tor_len, exit_len, n_test,  emb_size,  batch_size,device):

    file_fpath = pathlib.Path("baseline/PGD/PGDdeepcoffea_advsamples_time0.1396_size0.0272.p")
    with open(file_fpath, "rb") as fp:
        data = pickle.load(fp)
    adv_tor_windows,  original_exit_windows,time_l2_ratios,size_l2_ratios = data["adv_tor_windows"], data["original_exit_windows"],data["time_l2_ratios"],data["size_l2_ratios"]
    
    adv_tor_windows = torch.stack([torch.tensor(adv_tor_windows[i]) for i in range(len(adv_tor_windows))], dim=1)  # [,n_wins, tor_len*2]
    original_exit_windows = torch.stack([torch.tensor(original_exit_windows[i]) for i in range(len(original_exit_windows))], dim=1)  # [,n_wins, exit_len*2]
        
    state_dict = torch.load('target_model/deepcoffea/deepcoffea_d3_ws5_nw11_thr20_tl500_el800_nt1000_ap1e-01_es64_lr1e-03_mep100000_bs256/best_loss.pth', map_location=device)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    anchor = Deepcoffea.Model(emb_size=emb_size, input_size=tor_len*2).to(device)
    pandn = Deepcoffea.Model(emb_size=emb_size, input_size=exit_len*2).to(device)

    anchor.load_state_dict(state_dict['anchor_state_dict'])
    pandn.load_state_dict(state_dict['pandn_state_dict'])


    anchor.eval()
    pandn.eval()

    PGDtor_embs = []
    exit_embs = []
    
    #根据.p文件中的测试数量计算，选取了前五百条电路的样本
    n_test = adv_tor_windows.shape[1]  # Number of testing flow pairs
    with torch.no_grad():
        for i in tqdm(range(0, n_test, batch_size)):  

            PGDxa_batch = adv_tor_windows[:,i:i+batch_size,:]  # [ n_wins, batch_size,tor_len*2]
            xp_batch = original_exit_windows[:,i:i+batch_size,:]  # [ n_wins, batch_size, exit_len*2]
            PGDxa_batch = PGDxa_batch.reshape(-1, tor_len*2).float().to(device)   #Gxa_batch:[n_wins,batch_size, exit_len*2] -> [batch_size*n_wins, exit_len*2]
            xp_batch = xp_batch.reshape(-1, exit_len*2).float().to(device)   #xp_batch:[n_wins, batch_size ,exit_len*2] -> [batch_size*n_wins, exit_len*2]
            
            #PGDa_out: [batch_size*n_wins, emb_size], p_out: [batch_size*n_wins, emb_size]  
            PGDa_out = anchor(PGDxa_batch)
            p_out = pandn(xp_batch)
            
            PGDtor_embs.append(PGDa_out.cpu().numpy())
            exit_embs.append(p_out.cpu().numpy())
            

        PGDtor_embs = np.concatenate(PGDtor_embs) # (N, emb_size)
        exit_embs = np.concatenate(exit_embs)   # (N, emb_size)
        
        PGDcorr_matrix = cosine_similarity(PGDtor_embs, exit_embs)

        print('Time L2 distance rate: {0}'.format(time_l2_ratios))
        print('Size L2 distance rate: {0}'.format(size_l2_ratios))
        
        np.savez_compressed("baseline/PGD/deepcoffea/PGDcorrmatrix.npz", corr_matrix=PGDcorr_matrix,time_l2_ratios = time_l2_ratios, size_l2_ratios = size_l2_ratios)
                

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Coffea.")
    parser.add_argument("--delta", default=3, type=float, help="For window partition (see data_utils.py).")
    parser.add_argument("--win_size", default=5, type=float, help="For window partition (see data_utils.py).")
    parser.add_argument("--n_wins", default=11, type=int, help="For window partition (see data_utils.py).")
    parser.add_argument("--threshold", default=20, type=int, help="For window partition (see data_utils.py).")
    parser.add_argument("--tor_len", default=500, type=int, help="Flow size for the tor pairs.")
    parser.add_argument("--exit_len", default=800, type=int, help="Flow size for the exit pairs.")
    parser.add_argument("--n_test", default=1000, type=int, help="Number of testing flow pairs.")
    parser.add_argument("--emb_size", default=64, type=int, help="Feature embedding size.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
    parser.add_argument("--device", default="cuda:1", type=str, help="gpu.")
    
    args = parser.parse_args()
    main( args.delta, args.win_size, args.n_wins, args.threshold, args.tor_len, args.exit_len, args.n_test,  args.emb_size,  args.batch_size,args.device)